"""Dreamer 4 world model for stable-worldmodel.

Self-contained implementation of the causal tokenizer and shortcut-forcing
dynamics model from "Training Agents Inside of Scalable World Models"
(Hafner, Yan & Lillicrap, 2025), adapted for the stable-worldmodel
encode / forward / get_cost interface.

Reference: https://arxiv.org/abs/2509.24527
Based on: https://github.com/nicklashansen/dreamer4

All architecture components are inlined so this module has **no dependency**
on the external dreamer4 repository — only PyTorch is required.
"""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



class Modality(IntEnum):
    LATENT = -1
    IMAGE = 0
    ACTION = 1
    PROPRIO = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7


@dataclass(frozen=True)
class TokenLayout:
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if self.n_latents > 0:
            parts.append(torch.full((self.n_latents,), int(Modality.LATENT), dtype=torch.int32))
        for m, n in self.segments:
            if n > 0:
                parts.append(torch.full((n,), int(m), dtype=torch.int32))
        return torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        idx, out = 0, {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out



def temporal_patchify(videos: torch.Tensor, patch: int) -> torch.Tensor:
    """(B,T,C,H,W) → (B,T,Np,Dp)"""
    B, T, C, H, W = videos.shape
    x = videos.reshape(B * T, C, H, W)
    cols = F.unfold(x, kernel_size=patch, stride=patch).transpose(1, 2).contiguous()
    return cols.reshape(B, T, cols.shape[1], cols.shape[2])


def temporal_unpatchify(patches: torch.Tensor, H: int, W: int, C: int, patch: int) -> torch.Tensor:
    """(B,T,Np,Dp) → (B,T,C,H,W)"""
    B, T, Np, Dp = patches.shape
    x = patches.reshape(B * T, Np, Dp).transpose(1, 2).contiguous()
    out = F.fold(x, output_size=(H, W), kernel_size=patch, stride=patch)
    return out.reshape(B, T, C, H, W)


def pack_bottleneck_to_spatial(z: torch.Tensor, *, n_spatial: int, k: int) -> torch.Tensor:
    """(B,T,L,Db) → (B,T,n_spatial,Db*k)  where L == n_spatial*k"""
    B, T, L, D = z.shape
    return z.view(B, T, n_spatial, k * D)


def unpack_spatial_to_bottleneck(z: torch.Tensor, *, k: int) -> torch.Tensor:
    """(B,T,n_spatial,Db*k) → (B,T,n_spatial*k,Db)"""
    B, T, S, DK = z.shape
    D = DK // k
    return z.view(B, T, S * k, D)



def sinusoid_table(n: int, d: int, base: float = 10000.0, device=None) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)
    k = torch.floor(i / 2.0)
    div = torch.exp(-(2.0 * k) / max(1.0, float(d)) * math.log(base))
    ang = pos * div
    return torch.where((i % 2) == 0, torch.sin(ang), torch.cos(ang))


def add_sinusoidal_positions(tokens: torch.Tensor, scale: bool) -> torch.Tensor:
    B, T, S, D = tokens.shape
    pos_t = sinusoid_table(T, D, device=tokens.device)
    pos_s = sinusoid_table(S, D, device=tokens.device)
    pos = pos_t[None, :, None, :] + pos_s[None, None, :, :]
    if scale:
        pos = pos * (1.0 / math.sqrt(D))
    return tokens + pos.to(dtype=tokens.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.scale / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc_in = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        return self.drop(self.fc_out(self.drop(u * F.silu(v))))


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        N, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        drop = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=is_causal)
        return self.out(y.transpose(1, 2).contiguous().view(N, L, D))



class SpaceSelfAttentionModality(nn.Module):
    def __init__(self, d_model: int, n_heads: int, modality_ids: torch.Tensor,
                 n_latents: int, mode: str, dropout: float):
        super().__init__()
        self.n_latents = n_latents
        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)
        S = int(modality_ids.numel())
        self.register_buffer("attn_mask", self._build_allow(S).unsqueeze(0).unsqueeze(0), persistent=False)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def _build_allow(self, S: int) -> torch.Tensor:
        device = self.modality_ids.device
        q_idx = torch.arange(S, device=device).unsqueeze(1)
        k_idx = torch.arange(S, device=device).unsqueeze(0)
        is_q_lat = q_idx < self.n_latents
        q_mod = self.modality_ids[q_idx]
        k_mod = self.modality_ids[k_idx]
        same_mod = q_mod == k_mod

        if self.mode == "encoder":
            return torch.where(is_q_lat, torch.ones(S, S, dtype=torch.bool, device=device), same_mod)
        if self.mode == "decoder":
            is_k_lat = k_idx < self.n_latents
            return torch.where(is_q_lat, is_k_lat, same_mod | is_k_lat)
        if self.mode == "wm_agent":
            return torch.ones(S, S, dtype=torch.bool, device=device)
        if self.mode == "wm_agent_isolated":
            is_q_agent = q_mod == int(Modality.AGENT)
            is_k_agent = k_mod == int(Modality.AGENT)
            allow = torch.ones(S, S, dtype=torch.bool, device=device)
            allow = torch.where(~is_q_agent, ~is_k_agent, allow)
            allow = torch.where(is_q_agent, is_k_agent, allow)
            return allow
        raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x.shape
        x_flat = x.reshape(B * T, S, D)
        mask = self.attn_mask.expand(B * T, 1, S, S)
        return self.attn(x_flat, attn_mask=mask).reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 latents_only: bool, n_latents: int):
        super().__init__()
        self.latents_only = latents_only
        self.n_latents = n_latents
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x.shape
        if self.latents_only:
            L = self.n_latents
            lat = x[:, :, :L].permute(0, 2, 1, 3).contiguous().view(B * L, T, D)
            out = self.attn(lat, is_causal=True).view(B, L, T, D).permute(0, 2, 1, 3).contiguous()
            x = x.clone()
            x[:, :, :L] = out
            return x
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
        return self.attn(x_flat, is_causal=True).view(B, S, T, D).permute(0, 2, 1, 3).contiguous()



class BlockCausalLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_latents, modality_ids, space_mode,
                 dropout, mlp_ratio, layer_index, time_every, latents_only_time):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)
        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttentionModality(d_model, n_heads, modality_ids, n_latents, space_mode, dropout)
        self.drop1 = nn.Dropout(dropout)
        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model, n_heads, dropout, latents_only_time, n_latents)
            self.drop2 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.space(self.norm1(x)))
        if self.do_time:
            x = x + self.drop2(self.time(self.norm2(x)))
        x = x + self.mlp(self.norm3(x))
        return x


class BlockCausalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, depth, n_latents, modality_ids,
                 space_mode, dropout, mlp_ratio, time_every, latents_only_time):
        super().__init__()
        self.layers = nn.ModuleList([
            BlockCausalLayer(d_model, n_heads, n_latents, modality_ids,
                             space_mode, dropout, mlp_ratio, i, time_every, latents_only_time)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MAEReplacer(nn.Module):
    def __init__(self, d_model: int, p_min: float = 0.0, p_max: float = 0.9):
        super().__init__()
        self.p_min, self.p_max = p_min, p_max
        self.mask_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patches: torch.Tensor):
        B, T, Np, D = patches.shape
        device = patches.device
        if self.p_min == 0.0 and self.p_max == 0.0:
            return patches, torch.zeros(B, T, Np, 1, device=device, dtype=torch.bool), \
                   torch.ones(B, T, 1, device=device, dtype=patches.dtype)
        p = torch.empty(B, T, device=device).uniform_(self.p_min, self.p_max)
        keep_prob = (1.0 - p).unsqueeze(-1)
        keep = torch.rand(B, T, Np, device=device) < keep_prob
        replaced = torch.where(keep.unsqueeze(-1), patches, self.mask_token.view(1, 1, 1, D).to(patches.dtype))
        return replaced, (~keep).unsqueeze(-1), keep_prob


class Encoder(nn.Module):
    def __init__(self, *, patch_dim, d_model, n_latents, n_patches, n_heads, depth,
                 d_bottleneck, dropout=0.0, mlp_ratio=4.0, time_every=1,
                 latents_only_time=True, mae_p_min=0.0, mae_p_max=0.9,
                 scale_pos_embeds=True):
        super().__init__()
        self.d_model, self.n_latents, self.n_patches = d_model, n_latents, n_patches
        self.scale_pos_embeds = scale_pos_embeds
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)
        layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        self.transformer = BlockCausalTransformer(
            d_model, n_heads, depth, n_latents, layout.modality_ids(),
            "encoder", dropout, mlp_ratio, time_every, latents_only_time)
        self.mae = MAEReplacer(d_model, mae_p_min, mae_p_max)
        self.latents = nn.Parameter(torch.empty(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)

    def forward(self, patches: torch.Tensor):
        B, T, Np, _ = patches.shape
        proj = self.patch_proj(patches)
        proj_masked, mae_mask, keep_prob = self.mae(proj)
        lat = self.latents.view(1, 1, self.n_latents, -1).expand(B, T, -1, -1)
        tokens = add_sinusoidal_positions(torch.cat([lat, proj_masked], dim=2), self.scale_pos_embeds)
        enc = self.transformer(tokens)
        z = torch.tanh(self.bottleneck_proj(enc[:, :, :self.n_latents]))
        return z, (mae_mask, keep_prob)


class Decoder(nn.Module):
    def __init__(self, *, d_bottleneck, d_model, n_heads, depth, n_latents, n_patches,
                 d_patch, dropout=0.0, mlp_ratio=4.0, time_every=1,
                 latents_only_time=True, scale_pos_embeds=True):
        super().__init__()
        self.n_latents, self.n_patches = n_latents, n_patches
        self.scale_pos_embeds = scale_pos_embeds
        self.up_proj = nn.Linear(d_bottleneck, d_model)
        self.patch_queries = nn.Parameter(torch.empty(n_patches, d_model))
        nn.init.normal_(self.patch_queries, std=0.02)
        self.patch_head = nn.Linear(d_model, d_patch)
        layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        self.transformer = BlockCausalTransformer(
            d_model, n_heads, depth, n_latents, layout.modality_ids(),
            "decoder", dropout, mlp_ratio, time_every, latents_only_time)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, L, _ = z.shape
        lat = torch.tanh(self.up_proj(z))
        qry = self.patch_queries.view(1, 1, self.n_patches, -1).expand(B, T, -1, -1)
        tokens = add_sinusoidal_positions(torch.cat([lat, qry], dim=2), self.scale_pos_embeds)
        x = self.transformer(tokens)
        return torch.sigmoid(self.patch_head(x[:, :, L:]))


class Tokenizer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, patches: torch.Tensor):
        z, (mae_mask, keep_prob) = self.encoder(patches)
        pred = self.decoder(z)
        return pred, mae_mask, keep_prob



class ActionEncoder(nn.Module):
    def __init__(self, d_model: int, action_dim: int = 16, hidden_mult: float = 2.0):
        super().__init__()
        hidden = int(d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.base, std=0.02)
        self.fc1 = nn.Linear(action_dim, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, actions: Optional[torch.Tensor], *,
                batch_time_shape: Optional[Tuple[int, int]] = None,
                act_mask: Optional[torch.Tensor] = None, as_tokens: bool = True):
        if actions is None:
            B, T = batch_time_shape  # type: ignore[misc]
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            x = actions.clamp(-1, 1)
            if act_mask is not None:
                x = x * act_mask
            out = self.fc2(F.silu(self.fc1(x))) + self.base
        return out[:, :, None, :] if as_tokens else out


class Dynamics(nn.Module):
    def __init__(self, *, d_model, d_bottleneck, d_spatial, n_spatial, n_register,
                 n_agent, n_heads, depth, k_max, action_dim=16, dropout=0.0, mlp_ratio=4.0,
                 time_every=4, space_mode="wm_agent_isolated", scale_pos_embeds=True):
        super().__init__()
        self.d_model, self.d_spatial = d_model, d_spatial
        self.n_spatial, self.n_register, self.n_agent = n_spatial, n_register, n_agent
        self.k_max = k_max
        self.scale_pos_embeds = scale_pos_embeds

        self.spatial_proj = nn.Linear(d_spatial, d_model)
        self.register_tokens = nn.Parameter(torch.empty(n_register, d_model))
        nn.init.normal_(self.register_tokens, std=0.02)
        self.action_encoder = ActionEncoder(d_model=d_model, action_dim=action_dim)

        self.num_step_bins = int(math.log2(k_max)) + 1
        self.step_embed = nn.Embedding(self.num_step_bins, d_model)
        self.signal_embed = nn.Embedding(k_max + 1, d_model)

        segments: list = [
            (Modality.ACTION, 1), (Modality.SHORTCUT_SIGNAL, 1),
            (Modality.SHORTCUT_STEP, 1), (Modality.SPATIAL, n_spatial),
            (Modality.REGISTER, n_register),
        ]
        if n_agent > 0:
            segments.append((Modality.AGENT, n_agent))

        layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.agent_slice = sl.get(Modality.AGENT, slice(0, 0))

        self.transformer = BlockCausalTransformer(
            d_model, n_heads, depth, 0, layout.modality_ids(),
            space_mode, dropout, mlp_ratio, time_every, False)

        self.flow_x_head = nn.Linear(d_model, d_spatial)
        nn.init.zeros_(self.flow_x_head.weight)
        nn.init.zeros_(self.flow_x_head.bias)

    def forward(self, actions, step_idxs, signal_idxs, packed_enc_tokens, *,
                act_mask=None, agent_tokens=None):
        B, T = packed_enc_tokens.shape[:2]
        spatial = self.spatial_proj(packed_enc_tokens)
        action_tok = self.action_encoder(actions, batch_time_shape=(B, T), act_mask=act_mask, as_tokens=True)
        reg = self.register_tokens.view(1, 1, self.n_register, self.d_model).expand(B, T, -1, -1)
        step_tok = self.step_embed(step_idxs.long())[:, :, None, :]
        sig_tok = self.signal_embed(signal_idxs.long())[:, :, None, :]

        toks = [action_tok, sig_tok, step_tok, spatial, reg]
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros(B, T, self.n_agent, self.d_model,
                                           device=spatial.device, dtype=spatial.dtype)
            toks.append(agent_tokens)

        tokens = add_sinusoidal_positions(torch.cat(toks, dim=2), self.scale_pos_embeds)
        x = self.transformer(tokens)
        x1_hat = self.flow_x_head(x[:, :, self.spatial_slice])
        h_t = x[:, :, self.agent_slice] if self.n_agent > 0 else None
        return x1_hat, h_t


def recon_loss_from_mae(pred: torch.Tensor, target: torch.Tensor,
                        mae_mask: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss on MAE-masked patches."""
    mask = mae_mask.float()
    diff = pred.float() - target.float()
    sq = diff.mul(diff) * mask
    denom = mask.sum().clamp_min(1.0) * diff.shape[-1]
    return sq.sum() / denom


def lpips_on_mae_recon(
    lpips_fn,
    pred: torch.Tensor,
    target: torch.Tensor,
    mae_mask: torch.Tensor,
    *,
    H: int, W: int, C: int, patch: int,
    subsample_frac: float = 1.0,
) -> torch.Tensor:
    """LPIPS perceptual loss on MAE-reconstructed images.

    Composes the full image: predicted patches where masked, ground truth
    where unmasked. Then computes LPIPS in [-1, 1] image space.

    Args:
        lpips_fn: LPIPS network (e.g. ``lpips.LPIPS(net='alex')``).
        pred: Predicted patches (B, T, Np, Dp).
        target: Target patches (B, T, Np, Dp).
        mae_mask: Boolean mask (B, T, Np, 1), True = masked (reconstructed).
        H, W, C, patch: Image and patch dimensions.
        subsample_frac: Fraction of frames to evaluate (saves memory).

    Returns:
        Scalar LPIPS loss.
    """
    # Use pred where masked, target where visible
    recon = torch.where(mae_mask, pred, target)
    recon_img = temporal_unpatchify(recon.float(), H, W, C, patch)
    tgt_img = temporal_unpatchify(target.float(), H, W, C, patch)

    # Subsample frames for efficiency
    if subsample_frac < 1.0:
        step = max(1, int(1.0 / subsample_frac))
        recon_img = recon_img[:, ::step]
        tgt_img = tgt_img[:, ::step]

    # LPIPS expects [-1, 1]
    recon_img = (recon_img.clamp(0, 1) * 2.0 - 1.0).float()
    tgt_img = (tgt_img.clamp(0, 1) * 2.0 - 1.0).float()

    B, T = recon_img.shape[:2]
    recon_flat = recon_img.reshape(B * T, C, H, W)
    tgt_flat = tgt_img.reshape(B * T, C, H, W)

    with torch.autocast(device_type="cuda", enabled=False):
        lp = lpips_fn(recon_flat, tgt_flat)
    return lp.mean()


def shortcut_forcing_loss(
    dynamics: Dynamics,
    z1: torch.Tensor,
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    k_max: int,
    self_fraction: float = 0.25,
    bootstrap: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Shortcut forcing objective (x-prediction with ramp weight).

    Args:
        dynamics: Dynamics model.
        z1: Clean packed latent targets (B, T, Sz, Dz).
        actions: (B, T, A) or None.
        act_mask: (B, T, A), (A,), or None.
        k_max: Max sampling steps (power of 2).
        self_fraction: Batch fraction for bootstrap.
        bootstrap: Enable bootstrap (self-consistency) term.

    Returns:
        (loss, aux_dict)
    """
    device = z1.device
    B, T = z1.shape[:2]
    emax = int(round(math.log2(k_max)))

    B_self = int(round(self_fraction * B)) if bootstrap else 0
    B_self = max(0, min(B - 1, B_self))
    B_emp = B - B_self

    # Step indices
    step_emp = torch.full((B_emp, T), emax, device=device, dtype=torch.long)
    if B_self > 0:
        step_self = torch.randint(0, max(1, emax), (B_self, T), device=device)
        step_full = torch.cat([step_emp, step_self], 0)
    else:
        step_self = torch.zeros((0, T), device=device, dtype=torch.long)
        step_full = step_emp

    # Signal levels (tau)
    K_full = (1 << step_full).long()
    u = torch.rand(B, T, device=device)
    j = torch.floor(u * K_full.float()).long()
    sigma = j.float() / K_full.float()
    sigma_idx = (j * (k_max // K_full.clamp(min=1))).long()

    # Corrupt
    z0 = torch.randn_like(z1)
    z_tilde = (1 - sigma)[..., None, None] * z0 + sigma[..., None, None] * z1

    # Ramp weight w(τ) = 0.9τ + 0.1
    w = 0.9 * sigma + 0.1

    z1_hat, _ = dynamics(actions, step_full, sigma_idx, z_tilde, act_mask=act_mask)

    # Flow loss (empirical rows)
    flow_per = (z1_hat[:B_emp].float() - z1[:B_emp].float()).pow(2).mean(dim=(2, 3))
    loss_emp = (flow_per * w[:B_emp]).mean()

    # Bootstrap loss (self rows)
    loss_self = boot_mse = torch.zeros((), device=device)

    if B_self > 0 and bootstrap:
        sig_s, sig_idx_s = sigma[B_emp:], sigma_idx[B_emp:]
        zt_s = z_tilde[B_emp:]
        d_s = 1.0 / K_full[B_emp:].float()
        d_h = d_s / 2.0
        step_h = step_self + 1

        act_s = actions[B_emp:] if actions is not None else None
        msk_s = act_mask[B_emp:] if (act_mask is not None and act_mask.dim() > 1) else act_mask

        z1_h1, _ = dynamics(act_s, step_h, sig_idx_s, zt_s, act_mask=msk_s)
        bp = (z1_h1.float() - zt_s.float()) / (1 - sig_s).clamp_min(1e-6)[..., None, None]
        zp = zt_s.float() + bp * d_h[..., None, None]

        sig_plus_idx = sig_idx_s + (k_max * d_h).long()
        z1_h2, _ = dynamics(act_s, step_h, sig_plus_idx, zp.to(zt_s.dtype), act_mask=msk_s)
        bp2 = (z1_h2.float() - zp) / (1 - sig_s - d_h).clamp_min(1e-6)[..., None, None]

        v_hat = (z1_hat[B_emp:].float() - zt_s.float()) / (1 - sig_s).clamp_min(1e-6)[..., None, None]
        v_tgt = ((bp + bp2) / 2).detach()

        boot_per = (1 - sig_s).pow(2) * (v_hat - v_tgt).pow(2).mean(dim=(2, 3))
        loss_self = (boot_per * w[B_emp:]).mean()
        boot_mse = boot_per.mean()

    loss = (loss_emp * B_emp + loss_self * B_self) / B
    aux = dict(flow_mse=flow_per.mean().detach(), bootstrap_mse=boot_mse.detach(),
               loss_emp=loss_emp.detach(), loss_self=loss_self.detach(),
               sigma_mean=sigma.mean().detach())
    return loss, aux



def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def two_hot(x: torch.Tensor, cfg) -> torch.Tensor:
    if x.ndim == 0:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 1:
        x = x.unsqueeze(-1)
    nb = cfg.wm.num_bins
    vmin, vmax = cfg.wm.vmin, cfg.wm.vmax
    bs = (vmax - vmin) / (nb - 1)
    x = torch.clamp(symlog(x), vmin, vmax)
    idx = (x - vmin) / bs
    lo = idx.floor().long().clamp(0, nb - 2)
    off = idx - lo
    out = torch.zeros(x.shape[0], nb, device=x.device, dtype=x.dtype)
    out.scatter_(1, lo, 1 - off)
    out.scatter_(1, lo + 1, off)
    return out


def two_hot_inv(logits: torch.Tensor, cfg) -> torch.Tensor:
    bins = torch.linspace(cfg.wm.vmin, cfg.wm.vmax, cfg.wm.num_bins, device=logits.device)
    return symexp((F.softmax(logits, dim=-1) * bins).sum(-1, keepdim=True))


class NormedLinear(nn.Linear):
    """Linear → LayerNorm → Mish (TD-MPC2 style)."""
    def __init__(self, in_f, out_f, act=None, dropout=0.0):
        super().__init__(in_f, out_f)
        self.ln = nn.LayerNorm(out_f)
        self.act = act if act is not None else nn.Mish()
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = super().forward(x)
        if self.drop:
            x = self.drop(x)
        return self.act(self.ln(x))


def _mlp(in_dim, hid_dim, out_dim, act=None, dropout=0.0):
    layers = [NormedLinear(in_dim, hid_dim, dropout=dropout), NormedLinear(hid_dim, hid_dim)]
    layers.append(NormedLinear(hid_dim, out_dim, act=act) if act else nn.Linear(hid_dim, out_dim))
    return nn.Sequential(*layers)


def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def make_tau_schedule(k_max: int, schedule: str = "shortcut", d: float = 0.25) -> dict:
    if schedule == "finest":
        K = k_max
    elif schedule == "shortcut":
        K = int(round(1.0 / d))
    else:
        raise ValueError(schedule)
    e = int(round(math.log2(K)))
    dt = 1.0 / K
    tau = [i / K for i in range(K)]
    tau_idx = [i * (k_max // K) for i in range(K)]
    return dict(K=K, e=e, dt=dt, tau=tau, tau_idx=tau_idx)



class Dreamer4(nn.Module):
    """Dreamer 4 world model for the stable-worldmodel library.

    Wraps a causal tokenizer (encoder + decoder) and a shortcut-forcing
    dynamics transformer into the standard ``encode`` / ``forward`` /
    ``get_cost`` interface expected by planners like ``CEMSolver``.

    Training has two phases (controlled externally):
      1. **Tokenizer** – ``tokenizer_loss(frames)``
      2. **Dynamics**  – ``dynamics_loss(frames, actions)``  (tokenizer frozen)

    Args:
        cfg: OmegaConf config with ``cfg.wm.*`` and ``cfg.action_dim``.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        H = cfg.get("image_size", 128)
        self.H = self.W = H
        self.C = 3
        self.patch_size = cfg.wm.get("patch", 4)
        n_patches = (H // self.patch_size) ** 2
        d_patch = self.patch_size ** 2 * self.C

        n_latents = cfg.wm.get("n_latents", 16)
        d_bottleneck = cfg.wm.get("d_bottleneck", 32)
        d_tok = cfg.wm.get("d_model_tok", 256)
        depth_tok = cfg.wm.get("depth_tok", 8)
        n_heads_tok = cfg.wm.get("n_heads_tok", 4)
        dropout = cfg.wm.get("dropout", 0.0)
        mlp_ratio = cfg.wm.get("mlp_ratio", 4.0)
        scale_pos = cfg.wm.get("scale_pos_embeds", False)

        enc = Encoder(
            patch_dim=d_patch, d_model=d_tok, n_latents=n_latents,
            n_patches=n_patches, n_heads=n_heads_tok, depth=depth_tok,
            d_bottleneck=d_bottleneck, dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=cfg.wm.get("time_every_tok", 1),
            mae_p_min=cfg.wm.get("mae_p_min", 0.0),
            mae_p_max=cfg.wm.get("mae_p_max", 0.9),
            scale_pos_embeds=scale_pos,
        )
        dec = Decoder(
            d_bottleneck=d_bottleneck, d_model=d_tok, n_heads=n_heads_tok,
            depth=depth_tok, n_latents=n_latents, n_patches=n_patches,
            d_patch=d_patch, dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=cfg.wm.get("time_every_tok", 1), scale_pos_embeds=scale_pos,
        )
        self.tokenizer = Tokenizer(enc, dec)

        pf = cfg.wm.get("packing_factor", 2)
        assert n_latents % pf == 0
        n_spatial = n_latents // pf
        d_spatial = d_bottleneck * pf
        k_max = cfg.wm.get("k_max", 8)
        action_dim = cfg.action_dim

        self.dynamics = Dynamics(
            d_model=cfg.wm.get("d_model_dyn", 512),
            d_bottleneck=d_bottleneck, d_spatial=d_spatial,
            n_spatial=n_spatial, n_register=cfg.wm.get("n_register", 4),
            n_agent=cfg.wm.get("n_agent", 0),
            n_heads=cfg.wm.get("n_heads_dyn", 4),
            depth=cfg.wm.get("depth_dyn", 8), k_max=k_max,
            action_dim=action_dim,
            dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=cfg.wm.get("time_every_dyn", 4),
            space_mode=cfg.wm.get("space_mode", "wm_agent_isolated"),
            scale_pos_embeds=scale_pos,
        )

        mlp_dim = cfg.wm.get("mlp_dim", 256)
        num_bins = cfg.wm.get("num_bins", 101)
        self.reward = _mlp(n_spatial * d_spatial + action_dim, mlp_dim, num_bins)
        self.reward.apply(_weight_init)
        nn.init.zeros_(self.reward[-1].weight)

        self.lpips_weight = float(cfg.wm.get("lpips_weight", 0.2))
        self.lpips_frac = float(cfg.wm.get("lpips_frac", 0.5))
        self.lpips_fn = None
        if self.lpips_weight > 0:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=cfg.wm.get("lpips_net", "alex"))
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
        self.n_latents = n_latents
        self.d_bottleneck = d_bottleneck
        self.packing_factor = pf
        self.n_spatial = n_spatial
        self.d_spatial = d_spatial
        self.k_max = k_max
        self.latent_dim = n_spatial * d_spatial
        self.sched = make_tau_schedule(
            k_max, cfg.wm.get("schedule", "shortcut"), cfg.wm.get("eval_d", 0.25))


    @property
    def _emax(self):
        return int(round(math.log2(self.k_max)))

    def _pack(self, z: torch.Tensor) -> torch.Tensor:
        return z.unflatten(-1, (self.n_spatial, self.d_spatial))

    def _flat(self, z: torch.Tensor) -> torch.Tensor:
        return z.flatten(-2)


    def freeze_tokenizer(self):
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

    def tokenizer_loss(self, frames: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Tokenizer MAE reconstruction + LPIPS loss.

        L(θ) = L_MSE(θ) + lpips_weight · L_LPIPS(θ)   [paper eq. 5]

        Args:
            frames: (B, T, C, H, W) float in [0, 1].

        Returns:
            (loss, aux) with aux containing mse, lpips, z_std, keep_prob.
        """
        patches = temporal_patchify(frames, self.patch_size)

        # Encode (single pass — also gives us z for z_std diagnostic)
        z, (mae_mask, keep_prob) = self.tokenizer.encoder(patches)
        pred = self.tokenizer.decoder(z)

        mse = recon_loss_from_mae(pred, patches, mae_mask)

        # LPIPS perceptual loss
        lp = torch.zeros((), device=frames.device)
        if self.lpips_fn is not None and self.lpips_weight > 0:
            lp = lpips_on_mae_recon(
                self.lpips_fn, pred, patches, mae_mask,
                H=self.H, W=self.W, C=self.C, patch=self.patch_size,
                subsample_frac=self.lpips_frac,
            )

        loss = mse + self.lpips_weight * lp

        # z_std diagnostic (how spread are the latent codes)
        z_std = z.float().std().detach().item()

        return loss, dict(
            mse=mse.detach(),
            lpips=lp.detach(),
            z_std=z_std,
            keep_prob=keep_prob.mean().detach(),
        )


    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frame sequence → packed latents (B,T,Sz,Dz).  Tokenizer must be frozen."""
        with torch.no_grad():
            patches = temporal_patchify(frames, self.patch_size)
            z, _ = self.tokenizer.encoder(patches)
            return pack_bottleneck_to_spatial(z, n_spatial=self.n_spatial, k=self.packing_factor)

    def dynamics_loss(self, frames, actions=None, act_mask=None, bootstrap=True):
        """Shortcut forcing loss.  Returns (loss, aux) where aux['z1'] has encoded latents."""
        z1 = self.encode_sequence(frames)
        loss, aux = shortcut_forcing_loss(
            self.dynamics, z1, actions, act_mask, self.k_max,
            self.cfg.wm.get("self_fraction", 0.25), bootstrap)
        aux["z1"] = z1
        return loss, aux


    def encode(self, obs_dict: dict) -> torch.Tensor:
        """Pixels → flat latent.  Handles arbitrary leading dims."""
        pixels = obs_dict["pixels"].to(next(self.parameters()).dtype)
        if pixels.shape[-1] == 3:
            pixels = pixels.movedim(-1, -3)
        lead = pixels.shape[:-3]
        flat = pixels.reshape(-1, *pixels.shape[-3:]).unsqueeze(1)
        patches = temporal_patchify(flat, self.patch_size)
        with torch.no_grad():
            z, _ = self.tokenizer.encoder(patches)
        zp = pack_bottleneck_to_spatial(z, n_spatial=self.n_spatial, k=self.packing_factor)
        return zp[:, 0].flatten(1).reshape(*lead, self.latent_dim)


    @torch.no_grad()
    def _sample_next(self, past_packed, actions=None, act_mask=None):
        """Sample next packed latent via shortcut forcing.  (B,t,Sz,Dz) → (B,Sz,Dz)."""
        dev, dt_ = past_packed.device, past_packed.dtype
        B, t = past_packed.shape[:2]
        K, e, dt = self.sched["K"], self.sched["e"], self.sched["dt"]
        tau, tau_idx = self.sched["tau"], self.sched["tau_idx"]

        z = torch.randn(B, 1, self.n_spatial, self.d_spatial, device=dev, dtype=dt_)
        step = torch.full((B, t + 1), self._emax, device=dev, dtype=torch.long)
        step[:, -1] = e
        sig = torch.full((B, t + 1), self.k_max - 1, device=dev, dtype=torch.long)

        if act_mask is not None and act_mask.dim() == 1:
            act_mask = act_mask.view(1, 1, -1).expand(B, t + 1, -1)

        for i in range(K):
            sig[:, -1] = tau_idx[i]
            seq = torch.cat([past_packed, z], dim=1)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                x1, _ = self.dynamics(
                    actions[:, :t + 1] if actions is not None else None,
                    step, sig, seq, act_mask=act_mask)
            denom = max(1e-4, 1.0 - tau[i])
            z = (z.float() + (x1[:, -1:].float() - z.float()) / denom * dt).to(dt_)
        return z[:, 0]


    def forward(self, z: torch.Tensor, action: torch.Tensor):
        """One-step prediction.  z:(B,D), action:(B,A) → (next_z, reward_logits)."""
        B, A = z.shape[0], action.shape[-1]
        past = self._pack(z).unsqueeze(1)
        a = torch.zeros(B, 2, A, device=z.device, dtype=z.dtype)
        a[:, 1] = action
        nxt = self._flat(self._sample_next(past, a))
        return nxt, self.reward(torch.cat([z, action], -1))

    def rollout(self, z: torch.Tensor, actions: torch.Tensor):
        """Multi-step autoregressive rollout.  → (z_seq, r_logits_seq)."""
        B, H, A = actions.shape
        dev, dt_ = z.device, z.dtype
        hist = [self._pack(z)]
        a_hist: list = [torch.zeros(B, A, device=dev, dtype=dt_)]
        rewards = []
        for t in range(H):
            past = torch.stack(hist, 1)
            a_seq = torch.stack(a_hist + [actions[:, t]], 1)
            nxt = self._sample_next(past, a_seq)
            rewards.append(self.reward(torch.cat([self._flat(hist[-1]), actions[:, t]], -1)))
            hist.append(nxt)
            a_hist.append(actions[:, t])
        return (torch.stack([self._flat(h) for h in hist[1:]], 1),
                torch.stack(rewards, 1))

    @torch.inference_mode()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Evaluate candidate trajectories.  Compatible with CEMSolver."""
        dev = action_candidates.device
        keys = list(self.cfg.wm.get("encoding", {}).keys()) or ["pixels"]
        z = self.encode({k: info_dict[k].to(dev) for k in keys})
        B, N, H, A = action_candidates.shape
        if z.ndim == 2:
            z = z.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        else:
            z = z.reshape(B * N, -1)
        _, rl = self.rollout(z, action_candidates.reshape(B * N, H, A))
        G, disc = 0.0, 1.0
        gamma = self.cfg.wm.get("discount", 0.99)
        for t in range(H):
            G = G + disc * two_hot_inv(rl[:, t], self.cfg)
            disc *= gamma
        return -G.squeeze(-1).reshape(B, N)

    @torch.inference_mode()
    def get_action(self, info_dict: dict, horizon: int = 1, prefix_actions=None):
        """Simple random-shooting action selection."""
        dev = next(self.parameters()).device
        keys = list(self.cfg.wm.get("encoding", {}).keys()) or ["pixels"]
        z = self.encode({k: info_dict[k].to(dev) for k in keys})
        if prefix_actions is not None:
            for t in range(prefix_actions.shape[1]):
                z, _ = self.forward(z, prefix_actions[:, t].to(dev))
        B, A, ns = z.shape[0], self.cfg.action_dim, 512
        cands = torch.rand(B, ns, horizon, A, device=dev) * 2 - 1
        ze = z.unsqueeze(1).expand(-1, ns, -1).reshape(B * ns, -1)
        _, rl = self.rollout(ze, cands.reshape(B * ns, horizon, A))
        G = two_hot_inv(rl[:, 0], self.cfg).reshape(B, ns)
        return cands[torch.arange(B, device=dev), G.argmax(1)].unsqueeze(1)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode flat latent → pixels (B,C,H,W) in [0,1]."""
        zp = self._pack(z).unsqueeze(1)
        zl = unpack_spatial_to_bottleneck(zp, k=self.packing_factor)
        patches = self.tokenizer.decoder(zl)
        return temporal_unpatchify(patches, self.H, self.W, self.C, self.patch_size)[:, 0].clamp(0, 1)