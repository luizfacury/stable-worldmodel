"""Dreamer 4 world model for stable-worldmodel.

Self-contained implementation of the causal tokenizer and shortcut-forcing
dynamics model from "Training Agents Inside of Scalable World Models"
(Hafner, Yan & Lillicrap, 2025), adapted for the stable-worldmodel
encode / forward / get_cost interface.

Reference: https://arxiv.org/abs/2509.24527
Based on: https://github.com/nicklashansen/dreamer4
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
    B, T, C, H, W = videos.shape
    x = videos.reshape(B * T, C, H, W)
    cols = F.unfold(x, kernel_size=patch, stride=patch).transpose(1, 2).contiguous()
    return cols.reshape(B, T, cols.shape[1], cols.shape[2])


def temporal_unpatchify(patches: torch.Tensor, H: int, W: int, C: int, patch: int) -> torch.Tensor:
    B, T, Np, Dp = patches.shape
    x = patches.reshape(B * T, Np, Dp).transpose(1, 2).contiguous()
    out = F.fold(x, output_size=(H, W), kernel_size=patch, stride=patch)
    return out.reshape(B, T, C, H, W)


def pack_bottleneck_to_spatial(z: torch.Tensor, *, n_spatial: int, k: int) -> torch.Tensor:
    B, T, L, D = z.shape
    return z.view(B, T, n_spatial, k * D)


def unpack_spatial_to_bottleneck(z: torch.Tensor, *, k: int) -> torch.Tensor:
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
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 n_kv_heads: Optional[int] = None, qk_norm: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        kv_dim = self.n_kv_heads * self.head_dim
        self.qkv = nn.Linear(d_model, d_model + 2 * kv_dim, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

    def forward(self, x: torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        N, L, D = x.shape
        kv_dim = self.n_kv_heads * self.head_dim
        qkv = self.qkv(x)
        q = qkv[..., :D]
        k = qkv[..., D:D + kv_dim]
        v = qkv[..., D + kv_dim:]
        q = q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        drop = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=is_causal)
        return self.out(y.transpose(1, 2).contiguous().view(N, L, D))


class SpaceSelfAttentionModality(nn.Module):
    def __init__(self, d_model: int, n_heads: int, modality_ids: torch.Tensor,
                 n_latents: int, mode: str, dropout: float,
                 n_kv_heads: Optional[int] = None, qk_norm: bool = False):
        super().__init__()
        self.n_latents = n_latents
        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)
        S = int(modality_ids.numel())
        self.register_buffer("attn_mask", self._build_allow(S).unsqueeze(0).unsqueeze(0), persistent=False)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout, n_kv_heads=n_kv_heads, qk_norm=qk_norm)

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
            # Paper: agent tokens attend to all modalities; no other
            # modality attends back to agent tokens.
            is_q_agent = q_mod == int(Modality.AGENT)
            is_k_agent = k_mod == int(Modality.AGENT)
            allow = torch.ones(S, S, dtype=torch.bool, device=device)
            # Non-agent queries cannot see agent keys.
            allow = torch.where(~is_q_agent, ~is_k_agent, allow)
            # Agent queries can see everything — leave those rows alone.
            return allow
        raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x.shape
        x_flat = x.reshape(B * T, S, D)
        mask = self.attn_mask.expand(B * T, 1, S, S)
        return self.attn(x_flat, attn_mask=mask).reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 latents_only: bool, n_latents: int,
                 n_kv_heads: Optional[int] = None, qk_norm: bool = False):
        super().__init__()
        self.latents_only = latents_only
        self.n_latents = n_latents
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout, n_kv_heads=n_kv_heads, qk_norm=qk_norm)

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
                 dropout, mlp_ratio, layer_index, time_every, latents_only_time,
                 n_kv_heads: Optional[int] = None, qk_norm: bool = False):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)
        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttentionModality(d_model, n_heads, modality_ids, n_latents, space_mode, dropout, n_kv_heads=n_kv_heads, qk_norm=qk_norm)
        self.drop1 = nn.Dropout(dropout)
        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model, n_heads, dropout, latents_only_time, n_latents, n_kv_heads=n_kv_heads, qk_norm=qk_norm)
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
                 space_mode, dropout, mlp_ratio, time_every, latents_only_time,
                 n_kv_heads: Optional[int] = None, qk_norm: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            BlockCausalLayer(d_model, n_heads, n_latents, modality_ids,
                             space_mode, dropout, mlp_ratio, i, time_every, latents_only_time,
                             n_kv_heads=n_kv_heads, qk_norm=qk_norm)
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
        if not self.training or self.p_min == 0.0 and self.p_max == 0.0:
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
                 time_every=4, space_mode="wm_agent_isolated", scale_pos_embeds=True,
                 n_kv_heads: Optional[int] = None, qk_norm: bool = False):
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
            space_mode, dropout, mlp_ratio, time_every, False,
            n_kv_heads=n_kv_heads, qk_norm=qk_norm)

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


def recon_loss_from_mae(pred, target, mae_mask):
    mask = mae_mask.float()
    diff = pred.float() - target.float()
    sq = diff.mul(diff) * mask
    denom = mask.sum().clamp_min(1.0) * diff.shape[-1]
    return sq.sum() / denom


def lpips_on_mae_recon(lpips_fn, pred, target, mae_mask, *,
                      H, W, C, patch, subsample_frac=1.0):
    recon = torch.where(mae_mask, pred, target)
    recon_img = temporal_unpatchify(recon.float(), H, W, C, patch)
    tgt_img = temporal_unpatchify(target.float(), H, W, C, patch)
    if subsample_frac < 1.0:
        step = max(1, int(1.0 / subsample_frac))
        recon_img = recon_img[:, ::step]
        tgt_img = tgt_img[:, ::step]
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
    agent_tokens: Optional[torch.Tensor] = None,
    ctx_noise: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    device = z1.device
    B, T = z1.shape[:2]
    emax = int(round(math.log2(k_max)))

    B_self = int(round(self_fraction * B)) if bootstrap else 0
    B_self = max(0, min(B - 1, B_self))
    B_emp = B - B_self

    step_emp = torch.full((B_emp, T), emax, device=device, dtype=torch.long)
    if B_self > 0:
        step_self = torch.randint(0, max(1, emax), (B_self, T), device=device)
        step_full = torch.cat([step_emp, step_self], 0)
    else:
        step_self = torch.zeros((0, T), device=device, dtype=torch.long)
        step_full = step_emp

    K_full = (1 << step_full).long()
    u = torch.rand(B, T, device=device)
    j = torch.floor(u * K_full.float()).long()
    sigma = j.float() / K_full.float()
    sigma_idx = (j * (k_max // K_full.clamp(min=1))).long()

    z0 = torch.randn_like(z1)
    z_tilde = (1 - sigma)[..., None, None] * z0 + sigma[..., None, None] * z1
    if ctx_noise > 0.0:
        z_tilde = (1.0 - ctx_noise) * z_tilde + ctx_noise * torch.randn_like(z_tilde)
    w = 0.9 * sigma + 0.1

    z1_hat, h_t_full = dynamics(
        actions, step_full, sigma_idx, z_tilde,
        act_mask=act_mask, agent_tokens=agent_tokens,
    )

    flow_per = (z1_hat[:B_emp].float() - z1[:B_emp].float()).pow(2).mean(dim=(2, 3))
    loss_emp = (flow_per * w[:B_emp]).mean()

    loss_self = boot_mse = torch.zeros((), device=device)

    if B_self > 0 and bootstrap:
        sig_s, sig_idx_s = sigma[B_emp:], sigma_idx[B_emp:]
        zt_s = z_tilde[B_emp:]
        d_s = 1.0 / K_full[B_emp:].float()
        d_h = d_s / 2.0
        step_h = step_self + 1

        act_s = actions[B_emp:] if actions is not None else None
        msk_s = act_mask[B_emp:] if (act_mask is not None and act_mask.dim() > 1) else act_mask
        agt_s = agent_tokens[B_emp:] if agent_tokens is not None else None

        z1_h1, _ = dynamics(act_s, step_h, sig_idx_s, zt_s, act_mask=msk_s, agent_tokens=agt_s)
        bp = (z1_h1.float() - zt_s.float()) / (1 - sig_s).clamp_min(1e-6)[..., None, None]
        zp = zt_s.float() + bp * d_h[..., None, None]

        sig_plus_idx = sig_idx_s + (k_max * d_h).long()
        z1_h2, _ = dynamics(act_s, step_h, sig_plus_idx, zp.to(zt_s.dtype), act_mask=msk_s, agent_tokens=agt_s)
        bp2 = (z1_h2.float() - zp) / (1 - sig_s - d_h).clamp_min(1e-6)[..., None, None]

        v_hat = (z1_hat[B_emp:].float() - zt_s.float()) / (1 - sig_s).clamp_min(1e-6)[..., None, None]
        v_tgt = ((bp + bp2) / 2).detach()

        boot_per = (1 - sig_s).pow(2) * (v_hat - v_tgt).pow(2).mean(dim=(2, 3))
        loss_self = (boot_per * w[B_emp:]).mean()
        boot_mse = boot_per.mean()

    loss = (loss_emp * B_emp + loss_self * B_self) / B
    aux = dict(
        flow_mse=flow_per.mean().detach(),
        bootstrap_mse=boot_mse.detach(),
        loss_emp=loss_emp.detach(),
        loss_self=loss_self.detach(),
        sigma_mean=sigma.mean().detach(),
        h_t=h_t_full,
    )
    return loss, aux


def symlog(x): return torch.sign(x) * torch.log1p(x.abs())
def symexp(x): return torch.sign(x) * (torch.exp(x.abs()) - 1)


def two_hot(x, cfg):
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


def two_hot_inv(logits, cfg):
    bins = torch.linspace(cfg.wm.vmin, cfg.wm.vmax, cfg.wm.num_bins, device=logits.device)
    return symexp((F.softmax(logits, dim=-1) * bins).sum(-1, keepdim=True))


def two_hot_batch(x: torch.Tensor, cfg) -> torch.Tensor:
    """Vectorized two_hot for arbitrary leading dimensions.

    x: (...) -> out: (..., num_bins).
    Equivalent to applying two_hot element-wise but avoids a Python loop,
    using scatter_ along the last dimension instead.
    """
    nb = cfg.wm.num_bins
    vmin, vmax = cfg.wm.vmin, cfg.wm.vmax
    bs = (vmax - vmin) / (nb - 1)
    x = torch.clamp(symlog(x), vmin, vmax)
    idx = (x - vmin) / bs
    lo = idx.floor().long().clamp(0, nb - 2)       # (...)
    off = (idx - lo.float()).unsqueeze(-1)           # (..., 1)
    lo  = lo.unsqueeze(-1)                           # (..., 1)
    out = torch.zeros(*x.shape, nb, device=x.device, dtype=x.dtype)
    out.scatter_(-1, lo, 1.0 - off)
    out.scatter_(-1, lo + 1, off)
    return out


class NormedLinear(nn.Linear):
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


class MTPRewardHead(nn.Module):
    """Multi-token prediction reward head.

    Trunk of two NormedLinear layers, then L+1 parallel linear output heads
    (one per MTP distance n=0..L). Each head outputs num_bins logits.
    Input: h (B, T, d_in). Output: (B, T, L+1, num_bins).
    """
    def __init__(self, d_in: int, hidden: int, num_bins: int, mtp_length: int):
        super().__init__()
        self.trunk = nn.Sequential(
            NormedLinear(d_in, hidden),
            NormedLinear(hidden, hidden),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, num_bins) for _ in range(mtp_length + 1)
        ])
        # Match the old reward head's zero-init on the final layer.
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.trunk(h)
        return torch.stack([head(z) for head in self.heads], dim=-2)


class MTPPolicyHead(nn.Module):
    """Multi-token prediction policy head.

    Diagonal Gaussian policy for continuous actions, sharing the same
    trunk + parallel-heads structure as ``MTPRewardHead`` but outputting
    action means. ``log_std`` is a learned per-action-dim parameter shared
    across MTP distances. Input: h (B, T, d_in). Output: means
    (B, T, L+1, action_dim).
    """
    def __init__(self, d_in: int, hidden: int, action_dim: int, mtp_length: int,
                 log_std_init: float = -1.0):
        super().__init__()
        self.trunk = nn.Sequential(
            NormedLinear(d_in, hidden),
            NormedLinear(hidden, hidden),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, action_dim) for _ in range(mtp_length + 1)
        ])
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.trunk(h)
        return torch.tanh(torch.stack([head(z) for head in self.heads], dim=-2))

    def nll_per(self, means: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Per-(t,n) NLL summed over action dims. Returns (B, T, L+1)."""
        inv_var = torch.exp(-2.0 * self.log_std)
        sq = (actions - means).pow(2) * inv_var
        return 0.5 * (sq + 2.0 * self.log_std + math.log(2.0 * math.pi)).sum(-1)

    def nll(self, means: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood under a diagonal Gaussian.
        means, actions: (B, T, L+1, A). Returns a scalar.
        """
        return self.nll_per(means, actions).mean()


class ValueHead(nn.Module):
    """Scalar value function V(s) with symexp twohot output.

    Used in Phase 3 imagination training to estimate discounted cumulative reward
    from the agent-token embedding h_t.  Architecture mirrors a single-head
    variant of MTPRewardHead.
    Input: h (..., d_in). Output: (..., num_bins).
    """

    def __init__(self, d_in: int, hidden: int, num_bins: int):
        super().__init__()
        self.trunk = nn.Sequential(
            NormedLinear(d_in, hidden),
            NormedLinear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, num_bins)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(h))


def make_tau_schedule(k_max, schedule="shortcut", d=0.25):
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
        d_model_dyn = cfg.wm.get("d_model_dyn", 512)
        n_agent = cfg.wm.get("n_agent", 0)

        self.dynamics = Dynamics(
            d_model=d_model_dyn,
            d_bottleneck=d_bottleneck, d_spatial=d_spatial,
            n_spatial=n_spatial, n_register=cfg.wm.get("n_register", 4),
            n_agent=n_agent,
            n_heads=cfg.wm.get("n_heads_dyn", 4),
            depth=cfg.wm.get("depth_dyn", 8), k_max=k_max,
            action_dim=action_dim,
            dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=cfg.wm.get("time_every_dyn", 4),
            space_mode=cfg.wm.get("space_mode", "wm_agent_isolated"),
            scale_pos_embeds=scale_pos,
            n_kv_heads=cfg.wm.get("n_kv_heads_dyn", None),
            qk_norm=cfg.wm.get("qk_norm", False),
        )

        # Task embedding (fed into agent tokens during Phase 3)
        self.num_tasks = int(cfg.wm.get("num_tasks", 1))
        self.task_embed = nn.Embedding(self.num_tasks, d_model_dyn)
        nn.init.normal_(self.task_embed.weight, std=0.02)

        # Goal projection: encodes a goal latent into agent-token space.
        self.goal_proj = nn.Linear(n_spatial * d_spatial, d_model_dyn) if n_agent > 0 else None

        # MTP reward head reading from transformer agent-token output (h_t)
        mlp_dim = cfg.wm.get("mlp_dim", 256)
        num_bins = cfg.wm.get("num_bins", 101)
        self.mtp_length = int(cfg.wm.get("mtp_length", 8))
        self.reward = MTPRewardHead(
            d_in=d_model_dyn,
            hidden=mlp_dim,
            num_bins=num_bins,
            mtp_length=self.mtp_length,
        )
        self.reward.apply(_weight_init)
        # Re-zero outputs after _weight_init (it overwrites the constructor's zero init)
        for h in self.reward.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        # MTP policy head — trained jointly with the reward head on agent-token h_t.
        self.policy = MTPPolicyHead(
            d_in=d_model_dyn,
            hidden=mlp_dim,
            action_dim=action_dim,
            mtp_length=self.mtp_length,
            log_std_init=float(cfg.wm.get("policy_log_std_init", -1.0)),
        )
        self.policy.apply(_weight_init)
        for h in self.policy.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        # Value head — Phase 3 imagination training: predicts V(s) via TD lambda-returns.
        self.value = ValueHead(d_model_dyn, mlp_dim, num_bins)
        self.value.apply(_weight_init)
        nn.init.zeros_(self.value.head.weight)
        nn.init.zeros_(self.value.head.bias)

        # Slow EMA target critic — lagged copy of value used to bootstrap λ-returns.
        # Prevents the fast value head from chasing its own targets (bootstrap divergence).
        self.value_target = ValueHead(d_model_dyn, mlp_dim, num_bins)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_target.requires_grad_(False)
        self.value_ema_decay = float(cfg.wm.get("value_ema_decay", 0.98))

        # Policy prior — frozen copy of the Phase 2 policy head used for the KL
        # regularisation term in the PMPO objective (eq. 11).  Weights are copied
        # from self.policy via freeze_policy_prior() at the start of Phase 3.
        self.policy_prior = MTPPolicyHead(
            d_in=d_model_dyn,
            hidden=mlp_dim,
            action_dim=action_dim,
            mtp_length=self.mtp_length,
            log_std_init=float(cfg.wm.get("policy_log_std_init", -1.0)),
        )
        self.policy_prior.apply(_weight_init)
        for h in self.policy_prior.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

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
        self.n_agent = n_agent
        self.d_model_dyn = d_model_dyn
        self.latent_dim = n_spatial * d_spatial
        self.sched = make_tau_schedule(
            k_max, cfg.wm.get("schedule", "shortcut"), cfg.wm.get("eval_d", 0.25))

        # Running mean-square accumulators for per-loss RMS normalization
        # (paper 3: "we normalize all loss terms by running estimates of their RMS").
        # Initialized to 1 so normalization is a no-op at the start of training.
        # persistent=False: not saved in checkpoints (they warm up quickly).
        self.register_buffer('_rms_dyn',          torch.ones(1), persistent=False)
        self.register_buffer('_rms_reward',       torch.ones(1), persistent=False)
        self.register_buffer('_rms_policy',       torch.ones(1), persistent=False)
        self.register_buffer('_rms_value',        torch.ones(1), persistent=False)
        self.register_buffer('_rms_imag_policy',  torch.ones(1), persistent=False)

    @property
    def _emax(self):
        return int(round(math.log2(self.k_max)))

    def _pack(self, z):
        return z.unflatten(-1, (self.n_spatial, self.d_spatial))

    def _flat(self, z):
        return z.flatten(-2)

    def _build_agent_tokens(self, task_ids: torch.Tensor,
                            goal_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """task_ids: (B, T) long -> (B, T, n_agent, d_model_dyn)."""
        emb = self.task_embed(task_ids)                         # (B, T, d_model)
        if goal_z is not None and self.goal_proj is not None:
            emb = emb + self.goal_proj(goal_z).unsqueeze(1)    # broadcast over T
        return emb.unsqueeze(2).expand(-1, -1, self.n_agent, -1)

    def freeze_tokenizer(self):
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

    def freeze_policy_prior(self):
        """Copy policy → policy_prior and freeze it (called once at Phase 3 start)."""
        self.policy_prior.load_state_dict(self.policy.state_dict())
        self.policy_prior.eval()
        for p in self.policy_prior.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_value_target(self):
        d = self.value_ema_decay
        for p_tgt, p_src in zip(self.value_target.parameters(), self.value.parameters()):
            p_tgt.data.mul_(d).add_((1.0 - d) * p_src.data)

    @torch.no_grad()
    def imagine_rollout(
        self,
        start_z: torch.Tensor,
        horizon: int,
        start_h: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        goal_z: Optional[torch.Tensor] = None,
    ) -> dict:
        """Policy-driven imagination rollout from a starting latent state.

        The entire method runs under torch.no_grad().  The caller (imagination_forward)
        re-runs policy(h_t_seq) and value(h_t_seq) with gradients to compute losses.

        Args:
          start_z:  (B, latent_dim)     flat encoded latent from real data
          horizon:  int                 H imagination steps
          start_h:  (B, d_model_dyn)    agent token at start frame; if None,
                                        bootstrapped via a dummy _sample_next call
          task_ids: (B,) long           task IDs; defaults to 0

        Returns dict with:
          h_t_seq:  (B, H, d_model_dyn)  agent tokens at imagined steps (detached)
          z_seq:    (B, H, latent_dim)   imagined latent states (flat, detached)
          r_logits: (B, H, num_bins)     reward head logits (detached)
          actions:  (B, H, action_dim)   policy-sampled actions (detached)
        """
        if self.n_agent == 0:
            raise ValueError("imagine_rollout requires n_agent >= 1")

        B   = start_z.shape[0]
        dev = start_z.device
        dt_ = start_z.dtype
        A   = self.cfg.action_dim

        def _agent_tokens(length: int) -> Optional[torch.Tensor]:
            if self.n_agent == 0:
                return None
            if task_ids is None:
                tids = self._default_task_ids(B, length, dev)
            elif task_ids.ndim == 1:
                tids = task_ids.unsqueeze(1).expand(-1, length)
            else:
                tids = task_ids[:, :length]
            return self._build_agent_tokens(tids, goal_z=goal_z)

        start_packed = self._pack(start_z)  # (B, n_spatial, d_spatial)

        if start_h is None:
            dummy_a = torch.zeros(B, 2, A, device=dev, dtype=dt_)
            _, h_prev = self._sample_next(
                start_packed.unsqueeze(1), dummy_a,
                agent_tokens=_agent_tokens(2), return_h_last=True,
            )
        else:
            h_prev = start_h.to(dev)

        hist:   list = [start_packed]
        a_hist: list = [torch.zeros(B, A, device=dev, dtype=dt_)]

        h_t_list:      list = []
        z_list:        list = []
        r_logits_list: list = []
        actions_list:  list = []

        for t in range(horizon):
            # Deterministic actions: PMPO D+/D- split is based on advantage (state
            # quality), so the NLL gradient must correlate with action quality.
            # Stochastic sampling makes grad(NLL) ≈ noise × J_θ, which is
            # independent of the D+/D- split → expected PMPO gradient ≈ 0.
            means = self.policy(h_prev.unsqueeze(1))[:, 0, 0]   # (B, A) n=0 MTP
            a_t   = means.clamp(-1, 1)

            actions_list.append(a_t)
            a_hist.append(a_t)

            past  = torch.stack(hist, 1)
            a_seq = torch.stack(a_hist, 1)
            agt   = _agent_tokens(past.shape[1] + 1)

            # _sample_next has its own @torch.no_grad() — dynamics is frozen
            z_new, h_new = self._sample_next(
                past, a_seq, agent_tokens=agt, return_h_last=True,
            )

            hist.append(z_new)
            h_t_list.append(h_new)
            z_list.append(z_new)

            r_logits_list.append(self.reward(h_new.unsqueeze(1))[:, 0, 0])  # (B, num_bins)

            h_prev = h_new

        return {
            "h_t_seq":  torch.stack(h_t_list, 1),
            "z_seq":    torch.stack([self._flat(z) for z in z_list], 1),
            "r_logits": torch.stack(r_logits_list, 1),
            "actions":  torch.stack(actions_list, 1),
        }

    def _rms_norm(self, loss: torch.Tensor, buf: torch.Tensor,
                  decay: float = 0.99) -> torch.Tensor:
        """Normalize loss by its running RMS (paper 3).

        Updates buf (running mean-square) only during training, then divides
        loss by sqrt(buf). The denominator is clamped to >= 1 so small losses
        are never amplified — only large losses are scaled down.
        """
        if self.training:
            buf.copy_(decay * buf + (1.0 - decay) * loss.detach().pow(2))
        return loss / buf.sqrt().clamp_min(1.0)

    def tokenizer_loss(self, frames: torch.Tensor):
        patches = temporal_patchify(frames, self.patch_size)
        z, (mae_mask, keep_prob) = self.tokenizer.encoder(patches)
        pred = self.tokenizer.decoder(z)

        patch_mean = patches.mean(-1, keepdim=True)
        non_white = (patch_mean < 0.9).float()
        weights = 1.0 + 9.0 * non_white

        diff_sq = (pred.float() - patches.float()).pow(2)
        weighted = diff_sq * weights
        mask = mae_mask.float()
        denom = (mask * weights).sum().clamp_min(1.0) * diff_sq.shape[-1]
        mse = (weighted * mask).sum() / denom

        lp = torch.zeros((), device=frames.device)
        if self.lpips_fn is not None and self.lpips_weight > 0:
            lp = lpips_on_mae_recon(
                self.lpips_fn, pred, patches, mae_mask,
                H=self.H, W=self.W, C=self.C, patch=self.patch_size,
                subsample_frac=self.lpips_frac,
            )

        loss = mse + self.lpips_weight * lp
        z_std = z.float().std().detach().item()
        return loss, dict(mse=mse.detach(), lpips=lp.detach(),
                          z_std=z_std, keep_prob=keep_prob.mean().detach())

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        self.tokenizer.eval()
        with torch.no_grad():
            patches = temporal_patchify(frames, self.patch_size)
            z, _ = self.tokenizer.encoder(patches)
            return pack_bottleneck_to_spatial(z, n_spatial=self.n_spatial, k=self.packing_factor)

    def dynamics_loss(self, frames, actions=None, act_mask=None,
                      bootstrap=True, task_ids=None, goal_z=None):
        """Shortcut forcing loss. If n_agent > 0 and task_ids is given, builds
        agent tokens from task embeddings; h_t is returned in aux.
        """
        z1 = self.encode_sequence(frames)

        agent_tokens = None
        if self.n_agent > 0 and task_ids is not None:
            agent_tokens = self._build_agent_tokens(task_ids, goal_z=goal_z)

        train_ctx_noise = float(self.cfg.wm.get("ctx_noise", 0.0)) if self.training else 0.0
        loss, aux = shortcut_forcing_loss(
            self.dynamics, z1, actions, act_mask, self.k_max,
            self.cfg.wm.get("self_fraction", 0.25), bootstrap,
            agent_tokens=agent_tokens,
            ctx_noise=train_ctx_noise,
        )
        aux["z1"] = z1
        return loss, aux

    def encode(self, obs_dict: dict) -> torch.Tensor:
        pixels = obs_dict["pixels"].to(next(self.parameters()).dtype)
        if pixels.shape[-1] == 3:
            pixels = pixels.movedim(-1, -3)
        lead = pixels.shape[:-3]
        flat = pixels.reshape(-1, *pixels.shape[-3:]).unsqueeze(1)
        patches = temporal_patchify(flat, self.patch_size)
        self.tokenizer.eval()
        with torch.no_grad():
            z, _ = self.tokenizer.encoder(patches)
        zp = pack_bottleneck_to_spatial(z, n_spatial=self.n_spatial, k=self.packing_factor)
        return zp[:, 0].flatten(1).reshape(*lead, self.latent_dim)

    @torch.no_grad()
    def _sample_next(self, past_packed, actions=None, act_mask=None,
                     agent_tokens=None, return_h_last=False):
        """Sample next packed latent. Optionally return h_t at the new frame."""
        dev, dt_ = past_packed.device, past_packed.dtype
        B, t = past_packed.shape[:2]
        K, e, dt = self.sched["K"], self.sched["e"], self.sched["dt"]
        tau, tau_idx = self.sched["tau"], self.sched["tau_idx"]

        # Context corruption (paper 3.2): mix Gaussian noise into past latent
        # tokens before passing them to the dynamics model. This prevents
        # error accumulation during multi-step imagination rollouts.
        ctx_noise = float(self.cfg.wm.get("ctx_noise", 0.1))
        if ctx_noise > 0 and t > 0:
            noise_past = torch.randn_like(past_packed)
            past_packed = (1.0 - ctx_noise) * past_packed + ctx_noise * noise_past

        z = torch.randn(B, 1, self.n_spatial, self.d_spatial, device=dev, dtype=dt_)
        step = torch.full((B, t + 1), self._emax, device=dev, dtype=torch.long)
        step[:, -1] = e
        # Past frames: signal index reflects their (slightly corrupted) clean state.
        # New frame starts at sigma=0 (pure noise) and iterates to sigma=1.
        past_sig_bin = max(0, min(self.k_max - 1, int((1.0 - ctx_noise) * self.k_max)))
        sig = torch.full((B, t + 1), past_sig_bin, device=dev, dtype=torch.long)

        if act_mask is not None and act_mask.dim() == 1:
            act_mask = act_mask.view(1, 1, -1).expand(B, t + 1, -1)

        h_last = None
        for i in range(K):
            sig[:, -1] = tau_idx[i]
            seq = torch.cat([past_packed, z], dim=1)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                x1, h_t = self.dynamics(
                    actions[:, :t + 1] if actions is not None else None,
                    step, sig, seq, act_mask=act_mask,
                    agent_tokens=agent_tokens[:, :t + 1] if agent_tokens is not None else None,
                )
            denom = max(1e-4, 1.0 - tau[i])
            z = (z.float() + (x1[:, -1:].float() - z.float()) / denom * dt).to(dt_)
            if return_h_last and h_t is not None:
                h_last = h_t[:, -1, 0]  # (B, d_model_dyn)

        if return_h_last:
            return z[:, 0], h_last
        return z[:, 0]

    def _default_task_ids(self, B: int, T: int, device) -> torch.Tensor:
        return torch.zeros(B, T, dtype=torch.long, device=device)

    def forward(self, z, action):
        """One-step prediction. Returns (next_z, reward_logits_n0)."""
        B, A = z.shape[0], action.shape[-1]
        past = self._pack(z).unsqueeze(1)
        a = torch.zeros(B, 2, A, device=z.device, dtype=z.dtype)
        a[:, 1] = action

        agent_tokens = None
        if self.n_agent > 0:
            tids = self._default_task_ids(B, 2, z.device)
            agent_tokens = self._build_agent_tokens(tids)

        nxt, h_last = self._sample_next(
            past, a, agent_tokens=agent_tokens, return_h_last=True,
        )
        # Reward logits at the new frame, n=0 MTP head
        if h_last is not None:
            r_logits = self.reward(h_last.unsqueeze(1))[:, 0, 0]   # (B, num_bins)
        else:
            r_logits = torch.zeros(B, self.cfg.wm.num_bins, device=z.device)
        return self._flat(nxt), r_logits

    def rollout(self, z, actions):
        """Multi-step rollout. Returns (z_seq, r_logits_seq) with shapes
        (B, H, latent_dim) and (B, H, num_bins)."""
        B, H, A = actions.shape
        dev, dt_ = z.device, z.dtype

        hist = [self._pack(z)]
        a_hist: list = [torch.zeros(B, A, device=dev, dtype=dt_)]
        r_logits_list = []

        agent_tokens_full = None
        if self.n_agent > 0:
            tids = self._default_task_ids(B, H + 1, dev)
            agent_tokens_full = self._build_agent_tokens(tids)

        for t in range(H):
            past = torch.stack(hist, 1)
            a_seq = torch.stack(a_hist + [actions[:, t]], 1)
            agt = agent_tokens_full[:, : past.shape[1] + 1] if agent_tokens_full is not None else None

            nxt, h_last = self._sample_next(
                past, a_seq, agent_tokens=agt, return_h_last=True,
            )

            if h_last is not None:
                r_logits_list.append(self.reward(h_last.unsqueeze(1))[:, 0, 0])
            else:
                r_logits_list.append(torch.zeros(B, self.cfg.wm.num_bins, device=dev))

            hist.append(nxt)
            a_hist.append(actions[:, t])

        return (
            torch.stack([self._flat(h) for h in hist[1:]], 1),
            torch.stack(r_logits_list, 1),
        )

    @torch.inference_mode()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
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
    def decode(self, z):
        zp = self._pack(z).unsqueeze(1)
        zl = unpack_spatial_to_bottleneck(zp, k=self.packing_factor)
        patches = self.tokenizer.decoder(zl)
        return temporal_unpatchify(patches, self.H, self.W, self.C, self.patch_size)[:, 0].clamp(0, 1)
