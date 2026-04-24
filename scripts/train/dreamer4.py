"""Training script for Dreamer 4 world model in stable-worldmodel.

Supports four training phases controlled by ``cfg.wm.phase``:
  1. **tokenizer** – Train the causal tokenizer (encoder + decoder) with
     masked autoencoding and reconstruction loss.
  2. **dynamics** – Freeze the tokenizer and train the dynamics model with
     the shortcut forcing objective (NO reward head — this is pure
     video prediction pretraining, matching Phase 1 of the paper).
  3. **agent_finetune** – Load the pretrained dynamics, add agent tokens
     (n_agent >= 1), and train the reward head via MTP on the
     transformer's agent-token outputs h_t, while continuing to apply
     the dynamics loss. Matches Phase 2 of the paper (Section 3.3).
  4. **imagination_training** – Freeze the policy as a prior, then train the
     policy and value heads jointly via PMPO and TD λ-returns on imagined
     rollouts.  Phase 2 losses continue on real data (paper footnote †).
     Matches Phase 3 of the paper (Section 3.3).

Usage:
    # Phase 1: tokenizer
    python scripts/train/dreamer4.py wm.phase=tokenizer

    # Phase 2: dynamics (requires a trained tokenizer checkpoint)
    python scripts/train/dreamer4.py wm.phase=dynamics

    # Phase 3: agent finetune (requires a trained dynamics checkpoint)
    python scripts/train/dreamer4.py wm.phase=agent_finetune

    # Phase 4: imagination training (requires an agent_finetune checkpoint)
    python scripts/train/dreamer4.py wm.phase=imagination_training

Reference:
    Hafner, Yan & Lillicrap. "Training Agents Inside of Scalable World Models."
    arXiv:2509.24527, 2025.
"""
import os
import time
os.environ['STABLEWM_HOME'] = '/snfs2/luizfacury/datasets'
os.environ['MUJOCO_GL'] = 'egl'
from functools import partial
from pathlib import Path

import h5py

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

from stable_worldmodel.wm.dreamer4 import Dreamer4, two_hot, two_hot_inv, two_hot_batch
from stable_worldmodel.wm.dreamer4.module import (
    tokenizer_forward,
    dynamics_forward,
    agent_finetune_forward,
    imagination_forward,
    get_img_preprocessor,
    get_column_normalizer,
)


# ---------------------------------------------------------------------------
# Dataset wrapper: adds goal_pixels and steps_remaining from episode metadata
# ---------------------------------------------------------------------------

class GoalLastFrameWrapper:
    """Wraps an HDF5Dataset to add goal_pixels and steps_remaining per clip.

    goal_pixels: (3, H, W) uint8 — last frame of the episode (hindsight goal)
    steps_remaining: (T,) float32 — steps until episode end for each frame
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        ep_idx, start = self.dataset.clip_indices[idx]
        ep_len = int(self.dataset.lengths[ep_idx])
        frameskip = self.dataset.frameskip
        num_steps = self.dataset.num_steps

        local_steps = start + np.arange(num_steps) * frameskip
        steps_remaining = (ep_len - 1 - local_steps).astype(np.float32)
        batch['steps_remaining'] = torch.from_numpy(steps_remaining)

        goal_global = int(self.dataset.offsets[ep_idx]) + ep_len - 1
        self.dataset._open()
        goal_np = self.dataset.h5_file['pixels'][goal_global]   # (H, W, 3) uint8
        batch['goal_pixels'] = torch.from_numpy(goal_np.copy()).permute(2, 0, 1)
        return batch

    def __getattr__(self, name):
        return getattr(self.dataset, name)


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
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path='./config', config_name='dreamer4')
def run(cfg):
    torch.set_float32_matmul_precision('high')

    phase = cfg.wm.get('phase', 'dynamics')
    assert phase in ('tokenizer', 'dynamics', 'agent_finetune', 'imagination_training'), (
        f'Unknown phase: {phase}'
    )

    # ── Data setup ────────────────────────────────────────────────────
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    if not encoding_keys:
        encoding_keys = ['pixels']

    keys_to_load = list(set(encoding_keys + ['action']))

    base_dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.wm.horizon + 1,
        keys_to_load=keys_to_load,
        cache_dir='/dev/shm',
        frameskip=cfg['frameskip'],
    )

    with open_dict(cfg):
        cfg.action_dim = base_dataset.get_dim('action')

    raw_actions = base_dataset.get_col_data('action')[:]
    valid = raw_actions[~np.isnan(raw_actions).any(axis=1)]
    if valid.max() > 1.01 or valid.min() < -1.01:
        logging.warning(
            f'Actions outside [-1,1] detected (min={valid.min():.2f}, max={valid.max():.2f}). '
            'Consider normalizing.'
        )

    transforms = []
    img_size = cfg.get('image_size', 128)
    if 'pixels' in encoding_keys:
        transforms.append(get_img_preprocessor('pixels', 'pixels', img_size))

    extra_keys = [k for k in encoding_keys if k != 'pixels']
    for key in extra_keys:
        transforms.append(get_column_normalizer(base_dataset, key, key))

    base_dataset.transform = spt.data.transforms.Compose(*transforms)

    wrapped_dataset = GoalLastFrameWrapper(base_dataset)

    train_set, val_set = spt.data.random_split(
        wrapped_dataset, [cfg.train_split, 1 - cfg.train_split]
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True,
    )

    # Paper §3.2: dynamics pretraining uses NO agent tokens. Override here so
    # the Dynamics module is constructed without agent-token capacity, keeping
    # the checkpoint fully compatible when agent_finetune loads it with strict=False.
    if phase == 'dynamics':
        with open_dict(cfg):
            cfg.wm.n_agent = 0

    model = Dreamer4(cfg)

    # ── Checkpoint loading per phase ──────────────────────────────────
    if phase == 'dynamics':
        tok_ckpt = cfg.wm.get('tokenizer_ckpt', None)
        if tok_ckpt is not None:
            logging.info(f'Loading tokenizer from {tok_ckpt}')
            ckpt = torch.load(tok_ckpt, map_location='cpu', weights_only=False)

            if isinstance(ckpt, torch.nn.Module):
                model.tokenizer.load_state_dict(ckpt.tokenizer.state_dict(), strict=False)
            else:
                tok_state = ckpt.get('model', ckpt)
                tok_state = {
                    k.replace('tokenizer.', ''): v
                    for k, v in tok_state.items()
                    if k.startswith('tokenizer.')
                } or tok_state
                model.tokenizer.load_state_dict(tok_state, strict=False)

            logging.info('Tokenizer loaded successfully')

        model.freeze_tokenizer()

    elif phase in ('agent_finetune', 'imagination_training'):
        # Load tokenizer first (frozen).
        tok_ckpt = cfg.wm.get('tokenizer_ckpt', None)
        if tok_ckpt is not None:
            logging.info(f'Loading tokenizer from {tok_ckpt}')
            ckpt = torch.load(tok_ckpt, map_location='cpu', weights_only=False)
            if isinstance(ckpt, torch.nn.Module):
                model.tokenizer.load_state_dict(ckpt.tokenizer.state_dict(), strict=False)
            else:
                tok_state = ckpt.get('model', ckpt)
                tok_state = {
                    k.replace('tokenizer.', ''): v
                    for k, v in tok_state.items()
                    if k.startswith('tokenizer.')
                } or tok_state
                model.tokenizer.load_state_dict(tok_state, strict=False)
        model.freeze_tokenizer()

        # Load dynamics from Phase 2 checkpoint. strict=False because
        # n_agent was 0 in Phase 2 and is >=1 now, so the agent-token
        # parameters are new.
        dyn_ckpt = cfg.wm.get('dynamics_ckpt', None)
        assert dyn_ckpt is not None, (
            f'{phase} phase requires cfg.wm.dynamics_ckpt'
        )
        logging.info(f'Loading dynamics from {dyn_ckpt}')
        ckpt = torch.load(dyn_ckpt, map_location='cpu', weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            missing, unexpected = model.dynamics.load_state_dict(
                ckpt.dynamics.state_dict(), strict=False,
            )
        else:
            dyn_state = ckpt.get('model', ckpt)
            dyn_state = {
                k.replace('dynamics.', ''): v
                for k, v in dyn_state.items()
                if k.startswith('dynamics.')
            } or dyn_state
            missing, unexpected = model.dynamics.load_state_dict(
                dyn_state, strict=False,
            )
        logging.info(
            f'Dynamics loaded. Missing keys (expected, new agent params): '
            f'{len(missing)}. Unexpected: {len(unexpected)}.'
        )

        if phase == 'imagination_training':
            # For Phase 3 we additionally need the Phase 2 policy/reward weights.
            # Load the full agent_finetune checkpoint (which is a Dreamer4 object).
            af_ckpt = cfg.wm.get('agent_finetune_ckpt', None)
            assert af_ckpt is not None, (
                'imagination_training phase requires cfg.wm.agent_finetune_ckpt'
            )
            logging.info(f'Loading agent_finetune checkpoint from {af_ckpt}')
            af = torch.load(af_ckpt, map_location='cpu', weights_only=False)
            if isinstance(af, torch.nn.Module):
                model.reward.load_state_dict(af.reward.state_dict(), strict=True)
                model.policy.load_state_dict(af.policy.state_dict(), strict=True)
                if hasattr(af, 'task_embed'):
                    model.task_embed.load_state_dict(af.task_embed.state_dict(), strict=True)
                logging.info('reward, policy, task_embed loaded from agent_finetune checkpoint.')
            else:
                logging.warning('agent_finetune_ckpt is not a nn.Module — skipping policy/reward load.')
            # Freeze policy as prior before any gradient updates.
            model.freeze_policy_prior()
            logging.info('policy_prior frozen (copy of Phase 2 policy).')

    # ── Optimizer + forward function per phase ───────────────────────
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
    elif phase == 'dynamics':
        forward_fn = partial(dynamics_forward, cfg=cfg)
        optim = {
            'dyn_opt': opt_cfg(r'model\.dynamics\..*', cfg.optimizer.lr),
        }
    elif phase == 'agent_finetune':
        forward_fn = partial(agent_finetune_forward, cfg=cfg)
        # Pretrained dynamics weights use a lower finetune LR; brand-new
        # parameters (reward head, policy head, task embedding) use the
        # full base LR so they train from scratch at an appropriate rate.
        ft_lr = float(cfg.wm.get('finetune_lr', cfg.optimizer.lr))
        base_lr = float(cfg.optimizer.lr)
        optim = {
            'dyn_opt':    opt_cfg(r'model\.dynamics\..*',   ft_lr),
            'reward_opt': opt_cfg(r'model\.reward\..*',     base_lr),
            'policy_opt': opt_cfg(r'model\.policy\..*',     base_lr),
            'task_opt':   opt_cfg(r'model\.task_embed\..*', base_lr),
        }
    else:  # imagination_training
        forward_fn = partial(imagination_forward, cfg=cfg)
        ft_lr   = float(cfg.wm.get('finetune_lr', cfg.optimizer.lr))
        base_lr = float(cfg.optimizer.lr)
        optim = {
            # policy and value heads: full base LR (new objectives)
            'policy_opt': opt_cfg(r'model\.policy\..*',    base_lr),
            'value_opt':  opt_cfg(r'model\.value\..*',     base_lr),
            # dynamics, reward, task_embed: lower ft LR (preserve Phase 2 capability)
            'dyn_opt':    opt_cfg(r'model\.dynamics\..*',  ft_lr),
            'reward_opt': opt_cfg(r'model\.reward\..*',    ft_lr),
            'task_opt':   opt_cfg(r'model\.task_embed\..*', ft_lr),
            # policy_prior is frozen — excluded from optimizer
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
        run_id = f'{subdir}_{phase}_{int(time.time())}'
        logger = WandbLogger(
            name=f'dreamer4_{phase}_{cfg.dataset_name}_{subdir}',
            project=cfg.wandb.project,
            resume=None,
            id=run_id,
            log_model=False,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))

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