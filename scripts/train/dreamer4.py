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
    """Phase 1: train the causal tokenizer with masked autoencoding + LPIPS."""
    pixels = batch['pixels'].to(next(self.model.parameters()).device)
    if pixels.dtype == torch.uint8:
        pixels = pixels.float() / 255.0

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
    """Phase 2: pure shortcut forcing. No reward head."""
    device = next(self.model.parameters()).device

    pixels = batch['pixels'].to(device)
    if pixels.dtype == torch.uint8:
        pixels = pixels.float() / 255.0

    actions = batch['action'].to(device).float()

    B, T_plus_1 = pixels.shape[:2]
    T = T_plus_1 - 1

    frames = pixels[:, :-1]

    A = actions.shape[-1]
    act_aligned = torch.zeros(B, T, A, device=device)
    act_aligned[:, 1:] = actions[:, :T - 1].clamp(-1, 1)

    do_bootstrap = cfg.wm.get('bootstrap', True)
    dyn_loss, dyn_aux = self.model.dynamics_loss(
        frames, actions=act_aligned, bootstrap=do_bootstrap,
    )

    action_shuffle_ratio = torch.tensor(0.0, device=device)
    if stage == 'fit':
        with torch.no_grad():
            perm = torch.randperm(B, device=device)
            shuf_loss, _ = self.model.dynamics_loss(
                frames, actions=act_aligned[perm], bootstrap=False,
            )
            action_shuffle_ratio = shuf_loss / dyn_loss.clamp_min(1e-8)

    self.log_dict(
        {
            f'{stage}/loss': dyn_loss,
            f'{stage}/dyn_loss': dyn_loss,
            f'{stage}/flow_mse': dyn_aux['flow_mse'],
            f'{stage}/bootstrap_mse': dyn_aux['bootstrap_mse'],
            f'{stage}/sigma_mean': dyn_aux['sigma_mean'],
            f'{stage}/action_shuffle_ratio': action_shuffle_ratio,
        },
        on_step=True, sync_dist=False, prog_bar=True,
    )

    batch['loss'] = dyn_loss
    return batch


def agent_finetune_forward(self, batch, stage, cfg):
    """Phase 3 (paper's Phase 2): dynamics loss + MTP reward head on h_t.

    Adds agent tokens carrying task embeddings, reads reward from the
    transformer's agent-token output h_t, and trains with multi-token
    prediction of length L. The dynamics loss continues to run so the
    video prediction capability is preserved.
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
    L = int(cfg.wm.get('mtp_length', 8))

    frames = pixels[:, :-1]

    A = actions.shape[-1]
    act_aligned = torch.zeros(B, T, A, device=device)
    act_aligned[:, 1:] = actions[:, :T - 1].clamp(-1, 1)

    # Single-task PushT: task id 0 everywhere. Faithful to one-hot task
    # encoding with num_tasks=1.
    task_ids = torch.zeros(B, T, dtype=torch.long, device=device)

    # Dynamics loss (continues running per Section 3.3). h_t is returned
    # in aux so the reward head can read from it.
    do_bootstrap = cfg.wm.get('bootstrap', True)
    dyn_loss, dyn_aux = self.model.dynamics_loss(
        frames, actions=act_aligned, bootstrap=do_bootstrap,
        task_ids=task_ids,
    )

    # h_t: transformer output at the agent-token positions.
    h_t = dyn_aux['h_t']           # (B, T, n_agent, d_model)
    assert h_t is not None, (
        'h_t is None — did you set n_agent >= 1 in the config?'
    )
    h_flat = h_t[:, :, 0, :]       # (B, T, d_model); we use n_agent=1

    # Reward head returns (B, T, L+1, num_bins) via one output layer per
    # MTP distance.
    rew_logits = self.model.reward(h_flat)

    # Build MTP reward targets: for each time t, stack rewards[t+0..t+L],
    # clipping at the end of the sequence.
    t_idx = torch.arange(T, device=device)
    rew_targets = torch.stack(
        [rewards[:, torch.clamp(t_idx + n, max=T - 1)] for n in range(L + 1)],
        dim=2,
    )  # (B, T, L+1)

    # Valid mask: only include (t, n) pairs where t+n is within the sequence.
    # Without this, the last L timesteps have wrong targets (repeated last reward).
    n_range = torch.arange(L + 1, device=device)
    valid = (t_idx.unsqueeze(-1) + n_range.unsqueeze(0)) < T   # (T, L+1)
    valid = valid.unsqueeze(0).expand(B, -1, -1).float()        # (B, T, L+1)
    valid_count = valid.sum().clamp_min(1.0)

    # Symexp two-hot encoding — vectorized over all (B, T, L+1) at once.
    target_2h = two_hot_batch(rew_targets, cfg)   # (B, T, L+1, num_bins)

    log_probs = torch.log_softmax(rew_logits, dim=-1)
    reward_loss = (-(target_2h * log_probs).sum(-1) * valid).sum() / valid_count

    # Policy head: MTP NLL loss (diagonal Gaussian) over actions.
    # raw_actions[t] = action taken at frame t (policy target, not dynamics input).
    raw_actions = actions[:, :T].clamp(-1, 1)              # (B, T, A)
    act_targets = torch.stack(
        [raw_actions[:, torch.clamp(t_idx + n, max=T - 1)] for n in range(L + 1)],
        dim=2,
    )  # (B, T, L+1, A)

    policy_means = self.model.policy(h_flat)               # (B, T, L+1, A)
    policy_loss = (self.model.policy.nll_per(policy_means, act_targets) * valid).sum() / valid_count

    # Per-loss RMS normalization. Each term is divided by its own
    # running RMS, with denominator clamped to >= 1 so small losses are never
    # amplified. The buffers are updated only on the training step.
    dyn_n    = self.model._rms_norm(dyn_loss,    self.model._rms_dyn)
    reward_n = self.model._rms_norm(reward_loss, self.model._rms_reward)
    policy_n = self.model._rms_norm(policy_loss, self.model._rms_policy)
    total_loss = dyn_n + reward_n + policy_n

    with torch.no_grad():
        # Reward correlation on the n=0 MTP head.
        pred_r0 = two_hot_inv(rew_logits[:, :, 0, :], cfg).squeeze(-1)  # (B, T)
        gt_r0 = rewards[:, :T]
        pred_flat = pred_r0.reshape(-1).float()
        gt_flat = gt_r0.reshape(-1).float()
        pred_c = pred_flat - pred_flat.mean()
        gt_c = gt_flat - gt_flat.mean()
        denom = (pred_c.std() * gt_c.std()).clamp_min(1e-8)
        reward_corr = (pred_c * gt_c).mean() / denom
        # Action MAE on n=0 MTP head.
        action_mae = (policy_means[:, :, 0, :].detach() - raw_actions).abs().mean()

    self.log_dict(
        {
            f'{stage}/loss': total_loss,
            f'{stage}/dyn_loss': dyn_loss,
            f'{stage}/reward_loss': reward_loss,
            f'{stage}/reward_corr': reward_corr,
            f'{stage}/policy_loss': policy_loss,
            f'{stage}/action_mae': action_mae,
            f'{stage}/policy_log_std': self.model.policy.log_std.mean().detach(),
            f'{stage}/flow_mse': dyn_aux['flow_mse'],
            f'{stage}/bootstrap_mse': dyn_aux['bootstrap_mse'],
            f'{stage}/sigma_mean': dyn_aux['sigma_mean'],
            f'{stage}/rms_dyn':    self.model._rms_dyn.sqrt(),
            f'{stage}/rms_reward': self.model._rms_reward.sqrt(),
            f'{stage}/rms_policy': self.model._rms_policy.sqrt(),
        },
        on_step=True, sync_dist=False, prog_bar=True,
    )

    batch['loss'] = total_loss
    return batch


# ---------------------------------------------------------------------------
# Phase 3 helpers
# ---------------------------------------------------------------------------

def _kl_diagonal_gaussian(
    mu1: torch.Tensor, log_std1: torch.Tensor,
    mu2: torch.Tensor, log_std2: torch.Tensor,
) -> torch.Tensor:
    """KL(N(mu1, exp(log_std1)²) || N(mu2, exp(log_std2)²)), summed over last dim.

    Both mu tensors have shape (..., A).  log_std tensors may be (A,) scalars
    (shared across time/batch, as in MTPPolicyHead) or (..., A).
    Returns (...,).
    """
    var1 = (2.0 * log_std1).exp()
    var2 = (2.0 * log_std2).exp()
    kl = log_std2 - log_std1 + (var1 + (mu1 - mu2).pow(2)) / (2.0 * var2) - 0.5
    return kl.sum(-1)


def _lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute TD λ-returns from imagined rewards and bootstrapped values.

    Implements eq. 10 from the paper:
        R_H^λ = V(s_H)
        R_t^λ = r_t + γ[(1−λ)V(s_{t+1}) + λR_{t+1}^λ]   for t < H

    rewards, values: (B, H)  — 0-indexed, both referring to imagined step t+1
    Returns: (B, H)
    """
    B, H = rewards.shape
    returns = torch.empty_like(rewards)
    returns[:, -1] = values[:, -1]
    for t in range(H - 2, -1, -1):
        returns[:, t] = rewards[:, t] + gamma * (
            (1.0 - lam) * values[:, t + 1] + lam * returns[:, t + 1]
        )
    return returns


def imagination_forward(self, batch, stage, cfg):
    """Phase 3 (paper's Phase 3): PMPO + value head on imagined rollouts.

    Two loss streams are combined:
      A) Real-data losses (Phase 2 objectives) — preserve dynamics capability.
         Paper footnote †: dynamics + reward + policy (BC) losses continue.
      B) Imagination losses — policy PMPO + value TD λ-return.

    All imagination gradients flow only through the policy and value heads
    because imagine_rollout() and _sample_next() run under torch.no_grad().
    """
    device = next(self.model.parameters()).device

    gamma  = float(cfg.wm.get('imagination_discount', 0.997))
    lam    = float(cfg.wm.get('lambda_return',        0.95))
    alpha  = float(cfg.wm.get('pmpo_alpha',           0.5))
    beta   = float(cfg.wm.get('pmpo_beta',            0.3))
    H_imag = int(cfg.wm.get('imagination_horizon',    15))
    L      = int(cfg.wm.get('mtp_length',             8))

    # ── A) Real-data Phase 2 losses ───────────────────────────────────────
    pixels = batch['pixels'].to(device)
    if pixels.dtype == torch.uint8:
        pixels = pixels.float() / 255.0

    actions = batch['action'].to(device).float()
    rewards = batch['reward'].to(device).float()
    if rewards.ndim == 3 and rewards.shape[-1] == 1:
        rewards = rewards.squeeze(-1)

    B, T_plus_1 = pixels.shape[:2]
    T = T_plus_1 - 1
    frames = pixels[:, :-1]

    A = actions.shape[-1]
    act_aligned = torch.zeros(B, T, A, device=device)
    act_aligned[:, 1:] = actions[:, :T - 1].clamp(-1, 1)
    task_ids = torch.zeros(B, T, dtype=torch.long, device=device)

    do_bootstrap = cfg.wm.get('bootstrap', True)
    dyn_loss, dyn_aux = self.model.dynamics_loss(
        frames, actions=act_aligned, bootstrap=do_bootstrap, task_ids=task_ids,
    )

    h_t    = dyn_aux['h_t']          # (B, T, n_agent, d_model)
    h_flat = h_t[:, :, 0, :]         # (B, T, d_model)

    rew_logits = self.model.reward(h_flat)   # (B, T, L+1, num_bins)

    t_idx    = torch.arange(T, device=device)
    n_range  = torch.arange(L + 1, device=device)
    rew_tgts = torch.stack(
        [rewards[:, torch.clamp(t_idx + n, max=T - 1)] for n in range(L + 1)], dim=2,
    )  # (B, T, L+1)
    valid = ((t_idx.unsqueeze(-1) + n_range.unsqueeze(0)) < T
             ).unsqueeze(0).expand(B, -1, -1).float()
    valid_count = valid.sum().clamp_min(1.0)

    target_2h_rew = two_hot_batch(rew_tgts, cfg)
    reward_loss = (
        -(target_2h_rew * torch.log_softmax(rew_logits, dim=-1)).sum(-1) * valid
    ).sum() / valid_count

    raw_actions  = actions[:, :T].clamp(-1, 1)
    act_tgts     = torch.stack(
        [raw_actions[:, torch.clamp(t_idx + n, max=T - 1)] for n in range(L + 1)], dim=2,
    )  # (B, T, L+1, A)
    policy_means_real = self.model.policy(h_flat)    # (B, T, L+1, A)
    policy_loss_bc    = (
        self.model.policy.nll_per(policy_means_real, act_tgts) * valid
    ).sum() / valid_count

    # ── B) Imagination rollout ────────────────────────────────────────────
    # Start from a random real-data step so the imagination context is varied.
    with torch.no_grad():
        t_start  = int(torch.randint(0, max(1, T - H_imag), (1,)).item())
        start_z  = self.model._flat(dyn_aux['z1'][:, t_start])   # (B, latent_dim)
        start_h  = h_flat[:, t_start]                            # (B, d_model)

    task_1d = torch.zeros(B, dtype=torch.long, device=device)
    imag = self.model.imagine_rollout(
        start_z, horizon=H_imag, start_h=start_h, task_ids=task_1d,
    )

    h_imag   = imag['h_t_seq']   # (B, H, d_model) — detached
    r_logits = imag['r_logits']  # (B, H, num_bins) — detached
    a_imag   = imag['actions']   # (B, H, A)         — detached

    # Rewards from imagination (targets for λ-returns — no grad needed)
    with torch.no_grad():
        r_imag = two_hot_inv(r_logits, cfg).squeeze(-1)   # (B, H)

    # Value predictions — grad flows through value head params only
    val_logits = self.model.value(h_imag)                          # (B, H, num_bins)
    values     = two_hot_inv(val_logits, cfg).squeeze(-1)          # (B, H)

    # λ-returns are targets: stop grad so value head is not updated via λ
    with torch.no_grad():
        lambda_ret = _lambda_returns(r_imag, values.detach(), gamma, lam)  # (B, H)

    # ── Value loss: -log p_θ(R_t^λ | s_t) ────────────────────────────────
    target_2h_val = two_hot_batch(lambda_ret, cfg)                 # (B, H, num_bins)
    value_loss    = -(target_2h_val * torch.log_softmax(val_logits, dim=-1)).sum(-1).mean()

    # ── PMPO policy loss (eq. 11) ─────────────────────────────────────────
    # Policy means — grad flows through policy head params only
    policy_out        = self.model.policy(h_imag)       # (B, H, L+1, A)
    policy_means_imag = policy_out[:, :, 0, :]          # (B, H, A) — n=0 MTP

    # NLL of imagined actions under current policy
    policy_nll = self.model.policy.nll_per(
        policy_out[:, :, :1, :],     # (B, H, 1, A)
        a_imag.unsqueeze(2),         # (B, H, 1, A)
    )[:, :, 0]  # (B, H)

    # Advantages: sign determines which set each state belongs to
    advantages = (lambda_ret - values.detach()).detach()   # (B, H) — stop grad
    D_pos = advantages >= 0
    D_neg = ~D_pos
    n_pos = D_pos.float().sum().clamp_min(1.0)
    n_neg = D_neg.float().sum().clamp_min(1.0)

    pmpo_loss = (
        alpha / n_pos * policy_nll[D_pos].sum()           # reinforce good actions
        - (1.0 - alpha) / n_neg * policy_nll[D_neg].sum() # penalise bad actions
    )

    # KL regularisation towards policy prior — eq. 11 β·KL[π_θ || π_prior]
    # policy_prior is frozen so no grad flows through it
    prior_out   = self.model.policy_prior(h_imag)[:, :, 0, :]   # (B, H, A) no grad
    kl_per_step = _kl_diagonal_gaussian(
        policy_means_imag, self.model.policy.log_std,
        prior_out,         self.model.policy_prior.log_std,
    )  # (B, H)
    kl_loss = kl_per_step.mean()

    imag_policy_loss = pmpo_loss + beta * kl_loss

    # ── RMS normalization (paper §3) ──────────────────────────────────────
    dyn_n         = self.model._rms_norm(dyn_loss,         self.model._rms_dyn)
    reward_n      = self.model._rms_norm(reward_loss,      self.model._rms_reward)
    policy_bc_n   = self.model._rms_norm(policy_loss_bc,   self.model._rms_policy)
    value_n       = self.model._rms_norm(value_loss,       self.model._rms_value)
    imag_policy_n = self.model._rms_norm(imag_policy_loss, self.model._rms_imag_policy)
    total_loss    = dyn_n + reward_n + policy_bc_n + value_n + imag_policy_n

    # ── Diagnostics ───────────────────────────────────────────────────────
    with torch.no_grad():
        pred_r0     = two_hot_inv(rew_logits[:, :, 0, :], cfg).squeeze(-1)
        gt_r0       = rewards[:, :T]
        pred_c      = pred_r0.reshape(-1).float() - pred_r0.mean()
        gt_c        = gt_r0.reshape(-1).float() - gt_r0.mean()
        reward_corr = (pred_c * gt_c).mean() / (pred_c.std() * gt_c.std()).clamp_min(1e-8)
        action_mae  = (policy_means_real[:, :, 0, :].detach() - raw_actions).abs().mean()
        adv_pos_frac = D_pos.float().mean()
        mean_abs_adv = advantages.abs().mean()

    self.log_dict(
        {
            f'{stage}/loss':            total_loss,
            f'{stage}/dyn_loss':        dyn_loss,
            f'{stage}/reward_loss':     reward_loss,
            f'{stage}/reward_corr':     reward_corr,
            f'{stage}/policy_bc_loss':  policy_loss_bc,
            f'{stage}/action_mae':      action_mae,
            f'{stage}/value_loss':      value_loss,
            f'{stage}/pmpo_loss':       pmpo_loss,
            f'{stage}/kl_loss':         kl_loss,
            f'{stage}/imag_policy_loss': imag_policy_loss,
            f'{stage}/adv_pos_frac':    adv_pos_frac,
            f'{stage}/mean_abs_adv':    mean_abs_adv,
            f'{stage}/mean_value':      values.mean().detach(),
            f'{stage}/mean_return':     lambda_ret.mean(),
            f'{stage}/flow_mse':        dyn_aux['flow_mse'],
            f'{stage}/bootstrap_mse':   dyn_aux['bootstrap_mse'],
            f'{stage}/rms_dyn':         self.model._rms_dyn.sqrt(),
            f'{stage}/rms_reward':      self.model._rms_reward.sqrt(),
            f'{stage}/rms_policy':      self.model._rms_policy.sqrt(),
            f'{stage}/rms_value':       self.model._rms_value.sqrt(),
            f'{stage}/rms_imag_policy': self.model._rms_imag_policy.sqrt(),
        },
        on_step=True, sync_dist=False, prog_bar=True,
    )

    batch['loss'] = total_loss
    return batch


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_img_preprocessor(source, target, img_size=128):
    """Resize pixels to target size (NO ImageNet normalization)."""
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
    torch.set_float32_matmul_precision('high')

    phase = cfg.wm.get('phase', 'dynamics')
    assert phase in ('tokenizer', 'dynamics', 'agent_finetune', 'imagination_training'), (
        f'Unknown phase: {phase}'
    )

    # ── Data setup ────────────────────────────────────────────────────
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    if not encoding_keys:
        encoding_keys = ['pixels']

    keys_to_load = list(set(encoding_keys + ['action', 'reward']))

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