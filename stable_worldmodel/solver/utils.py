"""Shared utilities for planning solvers."""

import torch

from stable_worldmodel.protocols import Actionable


def build_init_action(
    model,
    info_dict: dict,
    init_action: torch.Tensor | None,
    horizon: int,
) -> torch.Tensor | None:
    """Extend or generate an initial action sequence using the model's actor.

    When the model implements the Actionable protocol, this function fills
    any missing planning steps by calling
    ``model.get_action(info_dict, horizon=remaining, prefix_actions=init_action)``,
    so the actor is rolled out from the latent state reached after applying
    the existing warm-start actions.
    If ``init_action`` already covers the full horizon it is returned unchanged.
    If the model is not Actionable, ``init_action`` is returned as-is (which may be None).

    Args:
        model: The solver's world model. Warm-starting is activated only when
            this model also implements the Actionable protocol.
        info_dict: Current observation dict with shape ``(n_envs, ...)``.
        init_action: Optional previous plan of shape ``(n_envs, t, action_dim)``
            where ``t <= horizon``.
        horizon: Full planning horizon expected by the solver.

    Returns:
        Action tensor of shape ``(n_envs, horizon, action_dim)``, or None if
        the model is not Actionable and no init_action was provided.
    """
    if not isinstance(model, Actionable):
        return init_action

    n_prev = init_action.shape[1] if init_action is not None else 0
    remaining = horizon - n_prev

    if remaining <= 0:
        return init_action

    with torch.no_grad():
        tail = model.get_action(
            info_dict, horizon=remaining, prefix_actions=init_action
        )
        # tail: (n_envs, remaining, action_dim)

    if init_action is not None:
        return torch.cat([init_action.to(tail.device), tail], dim=1)
    return tail
