"""Training script for Dreamer 4 world model in stable-worldmodel.

Supports two training phases controlled by ``cfg.wm.phase``:
  1. **tokenizer** – Train the causal tokenizer (encoder + decoder) with
     masked autoencoding and reconstruction loss.
  2. **dynamics** – Freeze the tokenizer and train the dynamics model with
     the shortcut forcing objective, plus an optional reward head.

Usage:
    # Phase 1: tokenizer
    python scripts/train/dreamer4.py wm.phase=tokenizer

    # Phase 2: dynamics (requires a trained tokenizer checkpoint)
    python scripts/train/dreamer4.py wm.phase=dynamics

Reference:
    Hafner, Yan & Lillicrap. "Training Agents Inside of Scalable World Models."
    arXiv:2509.24527, 2025.
"""
import os
from time import time
import time
os.environ['STABLEWM_HOME'] = '/snfs2/luizfacury/datasets'
os.environ['MUJOCO_GL'] = 'egl'
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from stable_worldmodel.wm.dreamer4 import Dreamer4, two_hot, two_hot_inv


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ModelObjectCallBack(Callback):
    """Periodically save the full model object for easy reloading."""

    def __init__(self, dirpath, filename='model_object', epoch_interval=1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
            path = self.dirpath / f'{self.filename}_epoch_{epoch}_object.ckpt'
            torch.save(pl_module.model, path)
            logging.info(f'Saved Dreamer4 model to {path}')


# ---------------------------------------------------------------------------
# Forward functions (one per phase)
# ---------------------------------------------------------------------------

def tokenizer_forward(self, batch, stage, cfg):
    """Phase 1: train the causal tokenizer with masked autoencoding + LPIPS.

    Expects ``batch['pixels']`` of shape (B, T+1, C, H, W) uint8 or float.
    """
    pixels = batch['pixels'].to(next(self.model.parameters()).device)
    if pixels.dtype == torch.uint8:
        pixels = pixels.float() / 255.0

    B, T = pixels.shape[:2]

    loss, aux = self.model.tokenizer_loss(pixels)

    psnr = 10.0 * torch.log10(1.0 / aux['mse'].clamp_min(1e-10))

    self.log_dict(
        {
            f'{stage}/loss': loss,
            f'{stage}/mse': aux['mse'],
            f'{stage}/lpips': aux['lpips'],
            f'{stage}/psnr': psnr,
            f'{stage}/keep_prob': aux['keep_prob'],
            f'{stage}/z_std': aux['z_std'],
        },
        on_step=True, sync_dist=False, prog_bar=True,
    )

    batch['loss'] = loss
    return batch


def dynamics_forward(self, batch, stage, cfg):
    """Phase 2: train dynamics with shortcut forcing + reward head.

    Expects ``batch`` with:
      - ``pixels``: (B, T+1, C, H, W)
      - ``action``:  (B, T+1, A) – first action is NaN or zero-padded
      - ``reward``:  (B, T+1) or (B, T+1, 1)
    """
    device = next(self.model.parameters()).device

    pixels = batch['pixels'].to(device)
    if pixels.dtype == torch.uint8:
        pixels = pixels.float() / 255.0

    actions = batch['action'].to(device).float()
    rewards = batch['reward'].to(device).float()
    if rewards.ndim == 3 and rewards.shape[-1] == 1:
        rewards = rewards.squeeze(-1)

    B, T_plus_1 = pixels.shape[:2]
    T = T_plus_1 - 1

    # Align: obs[0..T-1] are context, action[t] produced obs[t+1]
    frames = pixels[:, :-1]  # (B, T, C, H, W)

    # Shift actions: action[0] = zero, action[t] = action that led to frame[t]
    A = actions.shape[-1]
    act_aligned = torch.zeros(B, T, A, device=device)
    act_aligned[:, 1:] = actions[:, :T - 1].clamp(-1, 1)

    # --- Dynamics loss (shortcut forcing) ---
    do_bootstrap = cfg.wm.get('bootstrap', True)
    dyn_loss, dyn_aux = self.model.dynamics_loss(
        frames, actions=act_aligned, bootstrap=do_bootstrap,
    )

    # --- Reward loss (two-hot) ---
    reward_loss = torch.tensor(0.0, device=device)
    reward_coef = cfg.wm.get('reward_coef', 1.0)

    if reward_coef > 0:
        z_flat = dyn_aux['z1'].flatten(2).detach()  # (B, T, latent_dim)

        rew_target = rewards[:, :T]  # (B, T)
        z_a = torch.cat([z_flat, act_aligned], dim=-1)  # (B, T, latent_dim + A)
        rew_pred = self.model.reward(z_a)                # (B, T, num_bins)

        target_2h = torch.stack([two_hot(rew_target[:, t], cfg) for t in range(T)], dim=1)
        reward_loss = -(target_2h * torch.log_softmax(rew_pred, dim=-1)).sum(-1).mean()

    # --- Action shuffle ratio (diagnostic) ---
    action_shuffle_ratio = torch.tensor(0.0, device=device)
    if stage == 'train' and actions is not None:
        with torch.no_grad():
            perm = torch.randperm(B, device=device)
            shuf_loss, _ = self.model.dynamics_loss(
                frames, actions=act_aligned[perm], bootstrap=False,
            )
            action_shuffle_ratio = shuf_loss / dyn_loss.clamp_min(1e-8)

    total_loss = dyn_loss + reward_coef * reward_loss

    self.log_dict(
        {
            f'{stage}/loss': total_loss,
            f'{stage}/dyn_loss': dyn_loss,
            f'{stage}/reward_loss': reward_loss,
            f'{stage}/flow_mse': dyn_aux['flow_mse'],
            f'{stage}/bootstrap_mse': dyn_aux['bootstrap_mse'],
            f'{stage}/sigma_mean': dyn_aux['sigma_mean'],
            f'{stage}/action_shuffle_ratio': action_shuffle_ratio,
        },
        on_step=True, sync_dist=False, prog_bar=True,
    )

    batch['loss'] = total_loss
    return batch


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_img_preprocessor(source, target, img_size=128):
    """Resize pixels to target size (NO ImageNet normalization).

    Dreamer 4's tokenizer decoder uses sigmoid, so it reconstructs values
    in [0, 1].  Applying ImageNet normalization would shift targets to
    roughly [-2, +2.5], making reconstruction impossible.
    """
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(mean=[0, 0, 0], std=[1, 1, 1], source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


def get_column_normalizer(dataset, source, target):
    """Z-score normalizer for a dataset column."""
    data = torch.from_numpy(dataset.get_col_data(source)[:])
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).squeeze()
    std = data.std(0, keepdim=True).squeeze() + 1e-2

    def norm_fn(x):
        return ((x - mean.to(x.device)) / std.to(x.device)).float()

    return spt.data.transforms.WrapTorchTransform(norm_fn, source=source, target=target)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path='./config', config_name='dreamer4')
def run(cfg):
    """Train Dreamer 4 world model (tokenizer or dynamics phase).

    Args:
        cfg: Hydra configuration with ``cfg.wm.phase`` set to
             ``'tokenizer'`` or ``'dynamics'``.
    """
    torch.set_float32_matmul_precision('high')

    phase = cfg.wm.get('phase', 'dynamics')
    assert phase in ('tokenizer', 'dynamics'), f'Unknown phase: {phase}'

    # ── Data setup ────────────────────────────────────────────────────
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    if not encoding_keys:
        encoding_keys = ['pixels']

    keys_to_load = list(set(encoding_keys + ['action', 'reward']))

    base_dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.wm.horizon + 1,
        keys_to_load=keys_to_load,
        cache_dir=cfg.get('cache_dir'),
        frameskip=cfg['frameskip'],
    )

    with open_dict(cfg):
        cfg.action_dim = base_dataset.get_dim('action')

    # Validate actions are in [-1, 1]
    raw_actions = base_dataset.get_col_data('action')[:]
    valid = raw_actions[~np.isnan(raw_actions).any(axis=1)]
    if valid.max() > 1.01 or valid.min() < -1.01:
        logging.warning(
            f'Actions outside [-1,1] detected (min={valid.min():.2f}, max={valid.max():.2f}). '
            'Consider normalizing.'
        )

    # Transforms
    transforms = []
    img_size = cfg.get('image_size', 128)
    if 'pixels' in encoding_keys:
        transforms.append(get_img_preprocessor('pixels', 'pixels', img_size))

    extra_keys = [k for k in encoding_keys if k != 'pixels']
    for key in extra_keys:
        transforms.append(get_column_normalizer(base_dataset, key, key))

    base_dataset.transform = spt.data.transforms.Compose(*transforms)

    train_set, val_set = spt.data.random_split(
        base_dataset, [cfg.train_split, 1 - cfg.train_split]
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True,
    )

    model = Dreamer4(cfg)

    if phase == 'dynamics':
        tok_ckpt = cfg.wm.get('tokenizer_ckpt', None)
        if tok_ckpt is not None:
            logging.info(f'Loading tokenizer from {tok_ckpt}')
            ckpt = torch.load(tok_ckpt, map_location='cpu', weights_only=False)
            
            if isinstance(ckpt, torch.nn.Module):
                # saved as full model object via ModelObjectCallBack
                model.tokenizer.load_state_dict(ckpt.tokenizer.state_dict())
            else:
                # saved as state dict
                tok_state = ckpt.get('model', ckpt)
                tok_state = {
                    k.replace('tokenizer.', ''): v
                    for k, v in tok_state.items()
                    if k.startswith('tokenizer.')
                } or tok_state
                model.tokenizer.load_state_dict(tok_state, strict=False)
            
            logging.info('Tokenizer loaded successfully')

        model.freeze_tokenizer()

    def opt_cfg(module_regex, lr, eps=1e-8):
        oc = dict(cfg.optimizer)
        oc['lr'] = lr
        oc['eps'] = eps
        return {'modules': module_regex, 'optimizer': oc}

    if phase == 'tokenizer':
        forward_fn = partial(tokenizer_forward, cfg=cfg)
        optim = {
            'tok_opt': opt_cfg(r'model\.tokenizer\..*', cfg.optimizer.lr),
        }
    else:
        forward_fn = partial(dynamics_forward, cfg=cfg)
        optim = {
            'dyn_opt': opt_cfg(r'model\.dynamics\..*', cfg.optimizer.lr),
            'reward_opt': opt_cfg(r'model\.reward\..*', cfg.optimizer.lr),
        }

    module = spt.Module(
        model=model,
        forward=forward_fn,
        hparams=OmegaConf.to_container(cfg, resolve=True),
        optim=optim,
    )

    subdir = cfg.get('subdir') or f'dreamer4_{phase}'
    run_dir = Path(swm.data.utils.get_cache_dir(), subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enable:
        run_id = f'{subdir}_{phase}_{int(time.time())}'  # unique per run

        logger = WandbLogger(
            name=f'dreamer4_{phase}_{cfg.dataset_name}_{subdir}',
            project=cfg.wandb.project,
            resume=None,
            id=run_id,
            log_model=False,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Both phases have unused parameters (tokenizer phase doesn't touch
    # dynamics/reward; dynamics phase has a frozen tokenizer), so we need
    # find_unused_parameters=True for DDP.
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    strategy = trainer_kwargs.pop('strategy', 'ddp')
    if strategy in ('ddp', 'ddp_find_unused_parameters_false'):
        strategy = 'ddp_find_unused_parameters_true'

    trainer = pl.Trainer(
        **trainer_kwargs,
        strategy=strategy,
        logger=logger,
        callbacks=[
            ModelObjectCallBack(
                dirpath=run_dir,
                filename=f'dreamer4_{phase}',
            ),
        ],
    )

    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
    )()


if __name__ == '__main__':
    run()