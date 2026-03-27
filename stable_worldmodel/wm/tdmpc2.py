import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# ---------------------------------------------------------------------------
# TD-MPC2 Math Utilities
# ---------------------------------------------------------------------------


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    if x.ndim == 0:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 1:
        x = x.unsqueeze(-1)

    bin_size = (cfg.wm.vmax - cfg.wm.vmin) / (cfg.wm.num_bins - 1)
    x = torch.clamp(symlog(x), cfg.wm.vmin, cfg.wm.vmax)

    indices = (x - cfg.wm.vmin) / bin_size
    bin_idx = indices.floor().long()
    bin_offset = indices - bin_idx

    bin_idx = bin_idx.clamp(0, cfg.wm.num_bins - 2)

    soft_two_hot = torch.zeros(
        x.shape[0], cfg.wm.num_bins, device=x.device, dtype=x.dtype
    )
    soft_two_hot.scatter_(1, bin_idx, 1 - bin_offset)
    soft_two_hot.scatter_(1, bin_idx + 1, bin_offset)

    return soft_two_hot


def two_hot_inv(logits, cfg):
    device = logits.device
    bin_values = torch.linspace(
        cfg.wm.vmin, cfg.wm.vmax, cfg.wm.num_bins, device=device
    )
    probs = F.softmax(logits, dim=-1)
    x = torch.sum(probs * bin_values, dim=-1, keepdim=True)
    return symexp(x)


def log_std(x, low=-10, dif=12):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956
    return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


# ---------------------------------------------------------------------------
# TD-MPC2 Building Blocks
# ---------------------------------------------------------------------------


class SimNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # The paper uses V=8 as the default simplex dimensionality
        self.simplex_dim = cfg.wm.get('simnorm_dim', 8)

    def forward(self, x):
        shp = x.shape
        # Group the last dimension into L simplices of size V
        x = x.view(*shp[:-1], -1, self.simplex_dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class NormedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0., act=None):
        super().__init__(in_features, out_features, bias=bias)
        self.ln = nn.LayerNorm(out_features)
        self.act = act if act is not None else nn.Mish()
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))


class RunningScale(nn.Module):
    def __init__(self, tau=0.01):
        super().__init__()
        self.tau = tau
        self.register_buffer(
            'value', torch.ones(1, dtype=torch.float32) * 10.0
        )

    def update(self, x):
        with torch.no_grad():
            percentile_95 = torch.quantile(x.detach().float(), 0.95)
            percentile_05 = torch.quantile(x.detach().float(), 0.05)
            scale_val = torch.clamp(percentile_95 - percentile_05, min=1e-4)
            self.value.data.lerp_(scale_val.unsqueeze(0), self.tau)

    def forward(self, x, update=False):
        if update:
            self.update(x)
        return x / self.value


def mlp(in_dim, mlp_dim, out_dim, act=None, dropout=0.0):
    layers = [
        NormedLinear(in_dim, mlp_dim, dropout=dropout),
        NormedLinear(mlp_dim, mlp_dim),
    ]
    if act is not None:
        layers.append(NormedLinear(mlp_dim, out_dim, act=act))
    else:
        layers.append(nn.Linear(mlp_dim, out_dim))
    return nn.Sequential(*layers)


def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def zero_init(params):
    """Zero-initialize specific parameters."""
    for p in params:
        if p is not None:
            p.data.fill_(0)


class TDMPC2(nn.Module):
    """
    Main Neural Network Architecture for TD-MPC2.
    Handles dynamic encoding of modalities, latent dynamics, reward prediction, and action planning.

    Encoder takes observations only.

    Args:
        cfg: Configuration object containing model and training hyperparameters.
        extra_encoders: Optional pre-built ModuleDict of observation encoders.
            If provided, these are used directly instead of building default MLP
            encoders from cfg. Allows injecting custom encoder architectures
            (e.g. CNNs, transformers) without modifying this class.
            Output dims must match cfg.wm.encoding values.

    Assumptions:
        - Continuous Control: The algorithm assumes continuous action spaces.
        - Action Bounds: Actions are strictly assumed to be normalized to the range [-1.0, 1.0].
            The actor network and MPPI planner enforce this bound via Tanh and clamping.
        - Reward Scaling: Environment rewards and Q-values should fall roughly within the
            [vmin, vmax] range defined in the config, as they are discretized using two-hot encoding.
    """

    def __init__(self, cfg, extra_encoders: nn.ModuleDict | None = None):
        super().__init__()
        self.cfg = cfg
        self.scale = RunningScale(cfg.wm.tau)

        encoding_cfg = cfg.wm.get('encoding', {})
        self.use_pixels = 'pixels' in encoding_cfg
        self.latent_dim = 0

        if self.use_pixels:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 7, stride=2),
                nn.Mish(),
                nn.Conv2d(32, 32, 5, stride=2),
                nn.Mish(),
                nn.Conv2d(32, 32, 3, stride=2),
                nn.Mish(),
                nn.Conv2d(32, 32, 3, stride=1),
                nn.Mish(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
                cnn_out_dim = self.cnn(dummy).shape[1]

            pixel_dim = encoding_cfg['pixels']
            self.pixel_encoder = nn.Linear(cnn_out_dim, pixel_dim)
            self.latent_dim += pixel_dim

        if extra_encoders is not None:
            self.extra_encoders = extra_encoders
        else:
            # Default: build a two-layer MLP encoder for each non-pixel modality
            self.extra_encoders = nn.ModuleDict()
            for key, out_dim in encoding_cfg.items():
                if key == 'pixels':
                    continue
                in_dim = cfg.extra_dims[key]
                self.extra_encoders[key] = nn.Sequential(
                    NormedLinear(in_dim, cfg.wm.enc_dim),
                    nn.Linear(cfg.wm.enc_dim, out_dim),
                    nn.LayerNorm(out_dim),
                )

        # Accumulate latent dim from all non-pixel encoders
        for key, out_dim in encoding_cfg.items():
            if key != 'pixels':
                self.latent_dim += out_dim

        assert self.latent_dim > 0, (
            'Model must have pixels or at least one extra_encoder defined.'
        )

        self.sim_norm = SimNorm(cfg)

        # Latent dynamics model: predicts next latent state z' from (z, a)
        self.dynamics = mlp(
            self.latent_dim + cfg.action_dim, cfg.wm.mlp_dim,
            self.latent_dim, act=SimNorm(cfg),
        )

        # Reward predictor: predicts expected reward from (z, a) as a two-hot distribution
        self.reward = mlp(
            self.latent_dim + cfg.action_dim, cfg.wm.mlp_dim, cfg.wm.num_bins
        )

        # Policy prior (actor): outputs (mean, log_std) of a Gaussian over actions given z.
        # Used both to compute the policy loss and to warm-start CEM planning.
        self.pi = mlp(self.latent_dim, cfg.wm.mlp_dim, 2 * cfg.action_dim)

        # Ensemble of Q-functions: each predicts action-value from (z, a) as a two-hot
        # distribution. An ensemble is used for clipped double-Q to reduce overestimation.
        self.qs = nn.ModuleList(
            [
                mlp(
                    self.latent_dim + cfg.action_dim,
                    cfg.wm.mlp_dim,
                    cfg.wm.num_bins,
                    dropout=0.01,
                )
                for _ in range(cfg.wm.num_q)
            ]
        )
        self.target_qs = deepcopy(self.qs)
        for p in self.target_qs.parameters():
            p.requires_grad = False

        # Weight initialization (matches official TD-MPC2)
        self.apply(weight_init)
        zero_init([self.reward[-1].weight])
        for q in self.qs:
            zero_init([q[-1].weight])
        for q in self.target_qs:
            zero_init([q[-1].weight])

    def encode(self, obs_dict: dict) -> torch.Tensor:
        """Encode observations into a SimNorm-normalized latent state.

        Handles any number of leading dimensions without ndim checks, following
        the DINO-WM convention where the encoder owns the reshaping and callers
        pass observations as-is:

        - (B, dim)       → (B, latent_dim)          single-step inference
        - (B, T, dim)    → (B, T, latent_dim)        offline training sequence
        - (B, N, dim)    → (B, N, latent_dim)        CEM candidate expansion

        Args:
            obs_dict: Dictionary of observations with any leading shape.

        Returns:
            SimNorm-regularized latent state preserving all leading dimensions.
        """
        embeddings = []
        target_dtype = next(self.parameters()).dtype

        # Process primary vision modality — flatten all leading dims into batch
        if self.use_pixels:
            obs = obs_dict['pixels'].to(target_dtype)
            if obs.shape[-1] == 3:
                obs = obs.movedim(-1, -3)
            lead_dims = obs.shape[:-3]                      # e.g. (B,) or (B, T)
            obs_flat  = obs.reshape(-1, *obs.shape[-3:])    # (prod(lead), C, H, W)
            cnn_out   = self.cnn(obs_flat)
            z_pixels  = self.pixel_encoder(cnn_out).view(*lead_dims, -1)
            embeddings.append(z_pixels)

        # Process extra modalities (state, proprioception, etc.)
        for key, encoder in self.extra_encoders.items():
            obs      = obs_dict[key].to(target_dtype)       # (*lead, dim)
            lead     = obs.shape[:-1]
            obs_flat = obs.reshape(-1, obs.shape[-1])       # (prod(lead), dim)
            z        = encoder(obs_flat).view(*lead, -1)    # (*lead, enc_dim)
            embeddings.append(z)

        z_concat = torch.cat(embeddings, dim=-1)
        return self.sim_norm(z_concat)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step world model prediction.

        Given a latent state and action, predicts the next latent state via the
        dynamics model and the expected reward as a two-hot logit vector.

        Args:
            z: Current latent state of shape (B, latent_dim).
            action: Action of shape (B, action_dim).

        Returns:
            Tuple of (next_z, reward_logits) with shapes (B, latent_dim) and
            (B, num_bins) respectively.
        """
        z_a = torch.cat([z, action], dim=-1)
        return self.dynamics(z_a), self.reward(z_a)

    def rollout(
        self, z: torch.Tensor, horizon: int, num_trajs: int = 1
    ) -> torch.Tensor:
        """Roll out the actor policy from a latent state for a given horizon.

        Samples ``num_trajs`` stochastic trajectories and returns their mean.

        Args:
            z: Initial latent state of shape (B, latent_dim).
            horizon: Number of steps to unroll.
            num_trajs: Number of independent trajectories to average.

        Returns:
            Mean action sequence of shape (B, horizon, action_dim).
        """
        trajs = []
        for _ in range(num_trajs):
            curr_z, traj = z, []
            for _ in range(horizon):
                mean_raw, log_std_raw = self.pi(curr_z).chunk(2, dim=-1)
                act = torch.tanh(
                    mean_raw
                    + log_std(log_std_raw, low=-10, dif=12).exp()
                    * torch.randn_like(mean_raw)
                )
                traj.append(act)
                curr_z = self.dynamics(torch.cat([curr_z, act], dim=-1))
            trajs.append(torch.stack(traj, dim=1))  # (B, horizon, action_dim)
        return torch.stack(trajs).mean(0)            # (B, horizon, action_dim)

    def get_action(
        self,
        info_dict: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample an action sequence from the actor policy via latent rollout.

        Encodes the current observation into a latent state, optionally advances
        it through ``prefix_actions`` via the dynamics model, then calls
        ``rollout`` for ``horizon`` steps.

        Args:
            info_dict: Dictionary containing environment state information with
                shape (B, ...).
            horizon: Number of steps to plan.
            prefix_actions: Optional warm-start actions of shape
                (B, t, action_dim) with t < horizon. The latent state is
                advanced through these steps before the actor rollout.

        Returns:
            Action tensor of shape (B, horizon, action_dim).
        """
        device = next(self.parameters()).device
        encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())

        obs_dict = {key: info_dict[key].to(device) for key in encoding_keys}
        z = self.encode(obs_dict)

        if prefix_actions is not None:
            for t in range(prefix_actions.shape[1]):
                z = self.dynamics(
                    torch.cat([z, prefix_actions[:, t].to(device)], dim=-1)
                )

        num_trajs = self.cfg.wm.get('num_pi_trajs', 1)
        return self.rollout(z, horizon, num_trajs)  # (B, horizon, action_dim)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Evaluate the cost of candidate action trajectories.

        Rolls out the world model for each candidate, accumulates discounted
        rewards, and adds a terminal value estimate with an optional uncertainty
        penalty to favour conservative planning.

        Args:
            info_dict: Dictionary containing environment state with shape (B, N, ...).
            action_candidates: Candidate action sequences of shape (B, N, H, A).

        Returns:
            Cost tensor of shape (B, N). Lower is better.
        """
        device = action_candidates.device
        encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())

        obs_dict = {key: info_dict[key].to(device) for key in encoding_keys}
        z = self.encode(obs_dict)

        B, N, H, A = action_candidates.shape

        if z.ndim == 2 and z.shape[0] == B:
            z = z.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
        elif z.ndim == 3 and z.shape[0] == B and z.shape[1] == N:
            z = z.view(B * N, -1)
        elif z.ndim == 2 and z.shape[0] == B * N:
            pass
        else:
            raise ValueError(f'Unexpected latent state shape: {z.shape}')

        actions = action_candidates.view(B * N, H, A)

        G, discount = 0, 1.0
        c = self.cfg.wm.get('uncertainty_penalty', 0.5)
        termination = torch.zeros(B * N, 1, dtype=torch.float32, device=z.device)

        for t in range(H):
            z_a = torch.cat([z, actions[:, t]], dim=-1)
            reward = two_hot_inv(self.reward(z_a), self.cfg)
            z = self.dynamics(z_a)
            G = G + discount * (1 - termination) * reward
            discount = discount * self.cfg.wm.get('discount', 0.99)

        mu = self.pi(z).chunk(2, dim=-1)[0]
        action = torch.tanh(mu)
        z_a_term = torch.cat([z, action], dim=-1)

        q_logits = torch.stack([q(z_a_term) for q in self.qs])
        q_values  = torch.stack([two_hot_inv(logits, self.cfg) for logits in q_logits])

        q_mean = q_values.mean(dim=0)
        q_std  = q_values.std(dim=0)

        penalty        = c * q_mean.abs() * q_std
        conservative_q = q_mean - penalty
        total_return   = G + discount * (1 - termination) * conservative_q

        return -total_return.view(B, N)