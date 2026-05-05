"""Training forward functions and data helpers for Dreamer4.

These are the per-phase forward functions passed to spt.Module, plus
shared data pre-processing helpers.  Imported by scripts/train/dreamer4.py.
"""

import torch
import torch.nn.functional as F

import stable_pretraining as spt

from .dreamer4 import two_hot_batch, two_hot_inv


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
    goal_pixels = batch['goal_pixels'].to(device)                   # (B, 3, H, W) uint8
    if goal_pixels.dtype == torch.uint8:
        goal_pixels = goal_pixels.float() / 255.0
    img_size = int(cfg.get('image_size', 128))
    goal_pixels = F.interpolate(goal_pixels, size=(img_size, img_size),
                                mode='bilinear', align_corners=False)
    with torch.no_grad():
        goal_z = self.model.encode({'pixels': goal_pixels})         # (B, latent_dim)

    B, T_plus_1 = pixels.shape[:2]
    T = T_plus_1 - 1
    L = int(cfg.wm.get('mtp_length', 8))

    # Paper §3.3: scalar rewards r_t come directly from the dataset.
    rewards = batch['reward'][:, :T].to(device).float()             # (B, T)

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
        task_ids=task_ids, goal_z=goal_z,
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

    # Value head needs a wider bin range than the reward head: per-step rewards
    # fit in [vmin, vmax], but λ-returns accumulate over H steps and go well below.
    # We build a thin wrapper so reward-head calls keep the original bins.
    import types as _types
    _val_wm = _types.SimpleNamespace(**{
        **{k: getattr(cfg.wm, k) for k in dir(cfg.wm) if not k.startswith('_')},
        'vmin': float(cfg.wm.get('value_vmin', -3.0)),
        'vmax': float(cfg.wm.get('value_vmax', cfg.wm.vmax)),
    })
    val_cfg = _types.SimpleNamespace(wm=_val_wm)

    # ── A) Real-data Phase 2 losses ───────────────────────────────────────
    pixels = batch['pixels'].to(device)
    if pixels.dtype == torch.uint8:
        pixels = pixels.float() / 255.0

    actions = batch['action'].to(device).float()
    goal_pixels = batch['goal_pixels'].to(device)                   # (B, 3, H, W) uint8
    if goal_pixels.dtype == torch.uint8:
        goal_pixels = goal_pixels.float() / 255.0
    img_size = int(cfg.get('image_size', 128))
    goal_pixels = F.interpolate(goal_pixels, size=(img_size, img_size),
                                mode='bilinear', align_corners=False)
    with torch.no_grad():
        goal_z = self.model.encode({'pixels': goal_pixels})         # (B, latent_dim)

    B, T_plus_1 = pixels.shape[:2]
    T = T_plus_1 - 1

    # Paper §3.3: scalar rewards r_t come directly from the dataset.
    rewards = batch['reward'][:, :T].to(device).float()             # (B, T)

    frames = pixels[:, :-1]

    A = actions.shape[-1]
    act_aligned = torch.zeros(B, T, A, device=device)
    act_aligned[:, 1:] = actions[:, :T - 1].clamp(-1, 1)
    task_ids = torch.zeros(B, T, dtype=torch.long, device=device)

    do_bootstrap = cfg.wm.get('bootstrap', True)
    dyn_loss, dyn_aux = self.model.dynamics_loss(
        frames, actions=act_aligned, bootstrap=do_bootstrap, task_ids=task_ids,
        goal_z=goal_z,
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
        start_z, horizon=H_imag, start_h=start_h, task_ids=task_1d, goal_z=goal_z,
    )

    h_imag   = imag['h_t_seq']   # (B, H, d_model) — detached
    r_logits = imag['r_logits']  # (B, H, num_bins) — detached
    a_imag   = imag['actions']   # (B, H, A)         — detached

    # EMA-update the slow target before computing this step's targets.
    self.model.update_value_target()

    # Rewards from imagination (targets for λ-returns — no grad needed)
    with torch.no_grad():
        r_imag = two_hot_inv(r_logits, cfg).squeeze(-1)   # (B, H)

        # Slow target: used for bootstrap and advantage baseline.
        # The lagged weights break the feedback loop that drives value divergence.
        tgt_logits  = self.model.value_target(h_imag)
        values_tgt  = two_hot_inv(tgt_logits, val_cfg).squeeze(-1)        # (B, H)
        lambda_ret  = _lambda_returns(r_imag, values_tgt, gamma, lam)     # (B, H)

    # Fast value head — trained to fit lambda_ret; grad flows here only.
    val_logits = self.model.value(h_imag)                              # (B, H, num_bins)

    # ── Value loss: -log p_θ(R_t^λ | s_t) ────────────────────────────────
    target_2h_val = two_hot_batch(lambda_ret, val_cfg)                 # (B, H, num_bins)
    value_loss    = -(target_2h_val * torch.log_softmax(val_logits, dim=-1)).sum(-1).mean()

    # ── PMPO policy loss (eq. 11) ─────────────────────────────────────────
    policy_out        = self.model.policy(h_imag)       # (B, H, L+1, A)
    policy_means_imag = policy_out[:, :, 0, :]          # (B, H, A) — n=0 MTP

    policy_nll = self.model.policy.nll_per(
        policy_out[:, :, :1, :],     # (B, H, 1, A)
        a_imag.unsqueeze(2),         # (B, H, 1, A)
    )[:, :, 0]  # (B, H)

    # Advantages from slow target — stable baseline that doesn't chase itself.
    advantages = (lambda_ret - values_tgt).detach()   # (B, H) — no grad
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
            f'{stage}/mean_value_tgt':  values_tgt.mean().detach(),
            f'{stage}/mean_return':     lambda_ret.mean(),
            f'{stage}/r_imag_std':      r_imag.std().detach(),
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
