from functools import partial
from pathlib import Path
import random
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

from stable_worldmodel.wm.tdmpc2 import (
    TDMPC2,
    tdmpc2_forward,
)


class ModelObjectCallBack(Callback):
    """
    PyTorch Lightning callback to periodically save the entire model object to disk.
    """

    def __init__(self, dirpath, filename='model_object', epoch_interval=1):
        super().__init__()
        self.dirpath, self.filename, self.epoch_interval = (
            Path(dirpath),
            filename,
            epoch_interval,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
            path = self.dirpath / f'{self.filename}_epoch_{epoch}_object.ckpt'
            torch.save(pl_module.model, path)
            logging.info(f'Saved world model to {path}')



class GoalInjectTransform:
    """Injects the episode goal (last observation of the episode) into each sample.

    Uses a pre-computed table: goals_per_step[i] = last obs of the episode that
    contains timestep i.  Requires 'ep_offset' to be loaded so we can look up
    which episode a sequence belongs to.
    """

    def __init__(self, goals_per_step: torch.Tensor):
        self.goals_per_step = goals_per_step  # (N, obs_dim)

    def __call__(self, x: dict) -> dict:
        ep_offset = x['ep_offset']
        if isinstance(ep_offset, torch.Tensor):
            idx = int(ep_offset.flatten()[0].item())
        else:
            idx = int(ep_offset.flat[0])
        goal = self.goals_per_step[idx]        # (obs_dim,)
        T = ep_offset.shape[0]
        x['goal'] = goal.unsqueeze(0).expand(T, -1).clone()
        return x


def get_column_normalizer(dataset, source, target):
    """
    Creates a Z-score normalization transform for a specific dataset column.

    Args:
        dataset: The target dataset used to compute the global mean and standard deviation.
        source (str): The key of the input column to read.
        target (str): The key where the normalized output will be stored.

    Returns:
        Transform: A callable transform that applies the computed normalization to the data.
    """
    data = torch.from_numpy(dataset.get_col_data(source)[:])
    data = data[~torch.isnan(data).any(dim=1)]
    mean, std = (
        data.mean(0, keepdim=True).clone(),
        data.std(0, keepdim=True).clone(),
    )
    mean, std = mean.squeeze(), std.squeeze() + 1e-2

    def norm_fn(x):
        return ((x - mean.to(x.device)) / std.to(x.device)).float()

    return spt.data.transforms.WrapTorchTransform(
        norm_fn, source=source, target=target
    )


def get_img_preprocessor(source, target, img_size=64):
    """
    Creates a standardized image preprocessing pipeline.
    Applies ImageNet statistical normalization and resizes the images to the requested dimensions.

    Args:
        source (str): The dictionary key of the raw image input.
        target (str): The dictionary key to store the processed image output.
        img_size (int): The target height and width for the resizing operation.

    Returns:
        Transform: A composed image transformation pipeline.
    """
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )



@hydra.main(version_base=None, config_path='./config', config_name='tdmpc2')
def run(cfg):
    """
    Main training entry point for the TD-MPC2 model.

    Uses dataset rewards directly.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    torch.set_float32_matmul_precision('high')

    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    if not encoding_keys:
        raise ValueError('No encoding modalities defined in cfg.wm.encoding!')

    use_pixels = 'pixels' in encoding_keys
    use_goal = 'goal' in encoding_keys
    # 'goal' is computed from episode boundaries, not loaded directly from the dataset
    extra_keys = [k for k in encoding_keys if k not in ('pixels', 'goal')]

    keys_to_load = [k for k in encoding_keys if k != 'goal'] + ['action', 'reward']
    if use_goal:
        keys_to_load += ['ep_offset']

    base_dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.wm.horizon + 1,
        keys_to_load=keys_to_load,
        cache_dir=cfg.get('cache_dir'),
    )

    # Pre-compute goal for every timestep when goal conditioning is requested.
    # goal[i] = last observation of the episode containing timestep i,
    # derived from the ep_offset and ep_len metadata stored in the dataset.
    if use_goal:
        goal_obs_key = cfg.get('goal_obs_key')
        if goal_obs_key is None:
            raise ValueError(
                'cfg.goal_obs_key must specify which observation column to use as the '
                'goal (e.g. "state") when "goal" is present in cfg.wm.encoding.'
            )
        _raw_obs = base_dataset.get_col_data(goal_obs_key)[:]
        _ep_off = base_dataset.get_col_data('ep_offset')[:].flatten().astype(int)
        _ep_len = base_dataset.get_col_data('ep_len')[:].flatten().astype(int)
        _goal_idx = np.clip(_ep_off + _ep_len - 1, 0, len(_raw_obs) - 1)
        goals_per_step = torch.from_numpy(_raw_obs[_goal_idx]).float()
        logging.info(
            f'Goal conditioning enabled: goal = last obs of each episode '
            f'(source key: "{goal_obs_key}", dim: {goals_per_step.shape[-1]})'
        )

    raw_actions = base_dataset.get_col_data('action')[:]
    valid_actions = raw_actions[~np.isnan(raw_actions).any(axis=1)]
    act_max = valid_actions.max()
    act_min = valid_actions.min()

    if act_max > 1.01 or act_min < -1.01:
        logging.error(
            f'Dataset actions fall outside the [-1, 1] range! (Min: {act_min:.2f}, Max: {act_max:.2f}).\n'
            'TD-MPC2 uses a Tanh actor and strictly requires actions to be bounded between [-1, 1].\n'
            'Please normalize your dataset actions.'
        )
        raise ValueError(
            'Unnormalized actions detected in the dataset. Training aborted.'
        )

    with open_dict(cfg):
        cfg.action_dim = base_dataset.get_dim('action')
        cfg.extra_dims = {'action': cfg.action_dim}

        for key in extra_keys:
            cfg.extra_dims[key] = base_dataset.get_dim(key)

        if use_goal:
            cfg.extra_dims['goal'] = base_dataset.get_dim(goal_obs_key)

    transforms = []
    if use_pixels:
        transforms.append(
            get_img_preprocessor('pixels', 'pixels', cfg.image_size)
        )

    for key in extra_keys:
        transforms.append(get_column_normalizer(base_dataset, key, key))

    if use_goal:
        transforms.append(GoalInjectTransform(goals_per_step))
        _goal_clean = goals_per_step[~torch.isnan(goals_per_step).any(dim=1)]
        _g_mean = _goal_clean.mean(0).clone()
        _g_std = _goal_clean.std(0).clone() + 1e-2
        transforms.append(
            spt.data.transforms.WrapTorchTransform(
                lambda x, m=_g_mean, s=_g_std: ((x - m.to(x.device)) / s.to(x.device)).float(),
                source='goal',
                target='goal',
            )
        )

    base_dataset.transform = spt.data.transforms.Compose(*transforms)

    dataset = base_dataset

    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split]
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=True, persistent_workers=True
    )

    model = TDMPC2(cfg)

    def add_opt(module_regex, lr, eps=1e-8):
        opt_cfg = dict(cfg.optimizer)
        opt_cfg['lr'] = lr
        opt_cfg['eps'] = eps
        return {'modules': module_regex, 'optimizer': opt_cfg}

    module = spt.Module(
        model=model,
        forward=partial(tdmpc2_forward, cfg=cfg),
        hparams=OmegaConf.to_container(cfg, resolve=True),
        optim={
            'enc_opt': add_opt(
                r'model\.(cnn|pixel_encoder|extra_encoders|sim_norm).*',
                cfg.optimizer.lr * cfg.get('enc_lr_scale', 0.3),
            ),
            'wm_opt': add_opt(
                r'model\.(dynamics|reward|qs).*',
                cfg.optimizer.lr,
            ),
            'pi_opt': add_opt(r'model\.pi.*', cfg.optimizer.lr * 0.1, eps=1e-5),
        },
    )
    subdir = cfg.subdir
    run_dir = Path(swm.data.utils.get_cache_dir(), subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enable:
        logger = WandbLogger(
            name=f'{cfg.wm.name}_{cfg.dataset_name}_{subdir}',
            project=cfg.wandb.project,
            resume='allow' if subdir else None,
            id=subdir or None,
            log_model=False,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[
            ModelObjectCallBack(
                dirpath=run_dir, filename=cfg.output_model_name
            )
        ],
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
    )()


if __name__ == '__main__':
    run()
