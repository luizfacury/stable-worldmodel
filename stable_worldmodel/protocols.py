from typing import Any, Protocol, runtime_checkable
import numpy as np
import torch
import gymnasium as gym


class Costable(Protocol):
    """Protocol for world model cost functions."""

    def criterion(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Compute the cost criterion for action candidates.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        ...

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        """Compute cost for given action candidates based on info dictionary.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        ...


@runtime_checkable
class Solver(Protocol):
    """Protocol for model-based planning solvers."""

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        """Configure the solver with environment and planning specifications.

        Args:
            action_space: The action space of the environment.
            n_envs: Number of parallel environments.
            config: Planning configuration object.
        """
        ...

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        ...

    @property
    def n_envs(self) -> int:
        """Number of parallel environments being planned for."""
        ...

    @property
    def horizon(self) -> int:
        """Planning horizon length in timesteps."""
        ...

    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Solve the planning optimization problem to find optimal actions.

        Args:
            info_dict: Dictionary containing environment state information.
            init_action: Optional initial action sequence to warm-start the solver.

        Returns:
            Dictionary containing optimized actions and other solver-specific info.
        """
        ...


class Transformable(Protocol):
    """Protocol for reversible data transformations (e.g., normalizers, scalers)."""

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Apply preprocessing to input data.

        Args:
            x: Input data as a numpy array.

        Returns:
            Preprocessed data as a numpy array.
        """
        ...

    def inverse_transform(
        self, x: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """Reverse the preprocessing transformation.

        Args:
            x: Preprocessed data as a numpy array.

        Returns:
            Original data as a numpy array.
        """
        ...


class Actionable(Protocol):
    """Protocol for model action computation."""

    def get_action(
        info: dict, horizon: int = 1
    ) -> torch.Tensor:  # pragma: no cover
        """Compute action(s) from observation and goal.

        Args:
            info: Dictionary containing environment state information.
            horizon: Number of actions to return. When 1 (default), returns a
                single action of shape (..., action_dim). When > 1, returns an
                action sequence of shape (..., horizon, action_dim).

        Returns:
            A tensor of actions with shape (..., action_dim) if horizon == 1,
            or (..., horizon, action_dim) if horizon > 1.
        """
        ...
