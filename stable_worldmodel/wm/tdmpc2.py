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
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x):
        return F.mish(self.ln(super().forward(x)))


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


def mlp(in_dim, mlp_dim, out_dim, dropout=0.0):
    layers = [
        NormedLinear(in_dim, mlp_dim),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        NormedLinear(mlp_dim, mlp_dim),
        nn.Linear(mlp_dim, out_dim),
    ]
    return nn.Sequential(*layers)


class TDMPC2(nn.Module):
    """
    Main Neural Network Architecture for TD-MPC2.
    Handles dynamic encoding of modalities, latent dynamics, reward prediction, and action planning.

    Assumptions:
        - Continuous Control: The algorithm assumes continuous action spaces.
        - Action Bounds: Actions are strictly assumed to be normalized to the range [-1.0, 1.0].
            The actor network and MPPI planner enforce this bound via Tanh and clamping.
        - Reward Scaling: Environment rewards and Q-values should fall roughly within the
            [vmin, vmax] range defined in the config, as they are discretized using two-hot encoding.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.scale = RunningScale(cfg.wm.tau)

        # Determine modalities directly from the encoding dict
        encoding_cfg = cfg.wm.get('encoding', {})
        self.use_pixels = 'pixels' in encoding_cfg
        self.latent_dim = 0

        if self.use_pixels:
            self.cnn = nn.Sequential(
                nn.Conv2d(6, 32, 7, stride=2),
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
                dummy = torch.zeros(1, 6, cfg.image_size, cfg.image_size)
                cnn_out_dim = self.cnn(dummy).shape[1]

            pixel_dim = encoding_cfg['pixels']
            self.pixel_encoder = nn.Linear(cnn_out_dim, pixel_dim)
            self.latent_dim += pixel_dim

        self.extra_encoders = nn.ModuleDict()
        for key, out_dim in encoding_cfg.items():
            if key == 'pixels':
                continue  # Handled by primary backbone

            in_dim = cfg.extra_dims[key] * 2
            self.extra_encoders[key] = nn.Sequential(
                NormedLinear(in_dim, cfg.wm.enc_dim),
                NormedLinear(cfg.wm.enc_dim, out_dim),
            )
            self.latent_dim += out_dim

        assert self.latent_dim > 0, (
            'Model must have pixels or at least one extra_encoder defined.'
        )

        self.sim_norm = SimNorm(cfg)

        self.dynamics = nn.Sequential(
            NormedLinear(self.latent_dim + cfg.action_dim, cfg.wm.mlp_dim),
            NormedLinear(cfg.wm.mlp_dim, cfg.wm.mlp_dim),
            NormedLinear(cfg.wm.mlp_dim, self.latent_dim),
            SimNorm(cfg),
        )
        self.reward = mlp(
            self.latent_dim + cfg.action_dim, cfg.wm.mlp_dim, cfg.wm.num_bins
        )
        self.pi = mlp(self.latent_dim, cfg.wm.mlp_dim, 2 * cfg.action_dim)

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

    def encode(self, obs_dict, goal_dict):
        """
        Encodes primary vision and extra modalities separately, then concatenates them into a single latent state.

        Args:
            obs_dict (dict): Dictionary of current observations (e.g., 'pixels', 'state').
            goal_dict (dict): Dictionary of goal observations.

        Returns:
            torch.Tensor: The SimNorm-regularized latent representation of shape (B, latent_dim).
        """
        embeddings = []
        target_dtype = next(self.parameters()).dtype

        # Process Primary Vision
        if self.use_pixels:
            obs = obs_dict['pixels'].to(target_dtype)
            goal = goal_dict['pixels'].to(target_dtype)

            if obs.shape[-1] == 3:
                obs = obs.movedim(-1, -3)
                goal = goal.movedim(-1, -3)
            lead_dims = obs.shape[:-3]
            obs_flat = obs.reshape(-1, *obs.shape[-3:])
            goal_flat = goal.reshape(-1, *goal.shape[-3:])

            x_g = torch.cat([obs_flat, goal_flat], dim=-3)
            cnn_out = self.cnn(x_g)
            z_pixels = self.pixel_encoder(cnn_out).view(*lead_dims, -1)
            embeddings.append(z_pixels)

        # Process Extra Modalities (States, Proprio, etc.)
        for key, encoder in self.extra_encoders.items():
            obs = obs_dict[key].to(target_dtype)
            goal = goal_dict[key].to(target_dtype)
            x_g = torch.cat([obs, goal], dim=-1)
            z_extra = encoder(x_g)
            embeddings.append(z_extra)

        z_concat = torch.cat(embeddings, dim=-1)
        return self.sim_norm(z_concat)

    def forward(self, z, action):
        """
        Predicts the next latent state and expected reward given the current latent state and action.
        """
        z_a = torch.cat([z, action], dim=-1)
        return self.dynamics(z_a), self.reward(z_a)

    def rollout(
        self, z: torch.Tensor, horizon: int, num_trajs: int = 1
    ) -> torch.Tensor:
        """Roll out the actor policy from a latent state for a given horizon.

        Samples ``num_trajs`` stochastic trajectories and returns their mean.

        Args:
            z: Initial latent state of shape ``(B, latent_dim)``.
            horizon: Number of steps to unroll.
            num_trajs: Number of independent trajectories to average.

        Returns:
            Mean action sequence of shape ``(B, horizon, action_dim)``.
        """
        trajs = []
        for _ in range(num_trajs):
            curr_z, traj = z, []
            for _ in range(horizon):
                mean_raw, log_std_raw = self.pi(curr_z).chunk(2, dim=-1)
                act = torch.tanh(
                    mean_raw
                    + log_std_raw.clamp(-10, 2).exp()
                    * torch.randn_like(mean_raw)
                )
                traj.append(act)
                curr_z = self.dynamics(torch.cat([curr_z, act], dim=-1))
            trajs.append(torch.stack(traj, dim=1))  # (B, horizon, action_dim)
        return torch.stack(trajs).mean(0)  # (B, horizon, action_dim)

    def get_action(
        self,
        info_dict: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample an action sequence from the actor policy via latent rollout.

        Encodes the current observation/goal into a latent state, optionally
        advances it through ``prefix_actions`` via the dynamics model, then
        calls :meth:`rollout` for ``horizon`` steps.

        Args:
            info_dict: Dictionary containing environment state information with
                shape ``(B, ...)``.
            horizon: Number of steps to plan. Returns shape ``(B, action_dim)``
                when 1, or ``(B, horizon, action_dim)`` when > 1.
            prefix_actions: Optional warm-start actions of shape
                ``(B, t, action_dim)`` with ``t < horizon``. The latent state
                is advanced through these steps before the actor rollout.

        Returns:
            Action tensor of shape ``(B, action_dim)`` or ``(B, horizon, action_dim)``.
        """
        device = next(self.parameters()).device
        encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())

        obs_dict, goal_dict = {}, {}
        for key in encoding_keys:
            obs = info_dict[key].to(device)
            goal_key = f'goal_{key}' if f'goal_{key}' in info_dict else 'goal'
            goal = info_dict[goal_key].to(device)
            if key != 'pixels' and obs.ndim >= 3:
                obs = obs[..., -1, :]
                goal = goal[..., -1, :]
            obs_dict[key] = obs
            goal_dict[key] = goal

        z = self.encode(obs_dict, goal_dict)  # (B, latent_dim)

        if prefix_actions is not None:
            for t in range(prefix_actions.shape[1]):
                z = self.dynamics(
                    torch.cat([z, prefix_actions[:, t].to(device)], dim=-1)
                )

        num_trajs = self.cfg.wm.get('num_pi_trajs', 1)
        actions = self.rollout(
            z, horizon, num_trajs
        )  # (B, horizon, action_dim)

        if horizon == 1:
            return actions[:, 0]  # (B, action_dim)
        return actions

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """
        Evaluates the cost of candidate action trajectories.
        """
        device = action_candidates.device
        encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())

        obs_dict = {}
        goal_dict = {}

        for key in encoding_keys:
            obs = info_dict[key].to(device)

            eval_goal_key = f'goal_{key}'
            if eval_goal_key not in info_dict and 'goal' in info_dict:
                eval_goal_key = 'goal'
            goal = info_dict[eval_goal_key].to(device)

            if key != 'pixels':
                if obs.ndim >= 3:
                    obs = obs[..., -1, :]
                if goal.ndim >= 3:
                    goal = goal[..., -1, :]
            else:
                if (
                    obs.ndim >= 5
                ):  # e.g., (B, T, C, H, W) or (B, N, T, C, H, W)
                    obs = obs[..., -1, :, :, :]
                if goal.ndim >= 5:
                    goal = goal[..., -1, :, :, :]

            obs_dict[key] = obs
            goal_dict[key] = goal

        z = self.encode(obs_dict, goal_dict)

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

        termination = torch.zeros(
            B * N, 1, dtype=torch.float32, device=z.device
        )

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
        q_values = torch.stack(
            [two_hot_inv(logits, self.cfg) for logits in q_logits]
        )

        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)

        penalty = c * q_mean.abs() * q_std
        conservative_q = q_mean - penalty

        total_return = G + discount * (1 - termination) * conservative_q

        cost = -total_return.view(B, N)
        return cost
