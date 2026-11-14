"""Offline data: random-input trajectories from a plant."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectories:
    """``x[:, k+1] = f(x[:, k], u[:, k])`` for ``n`` trajectories of ``horizon`` steps."""

    x: np.ndarray  # (n, horizon + 1, n_x)
    u: np.ndarray  # (n, horizon, n_u)

    def transitions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flattened ``(x_i, u_i, x_{i+1})`` triplets."""
        n, h, _ = self.u.shape
        return (self.x[:, :-1].reshape(n * h, -1), self.u.reshape(n * h, -1), self.x[:, 1:].reshape(n * h, -1))


def collect(plant, n_traj: int, horizon: int, seed: int = 0, scale: float = 1.0) -> Trajectories:
    """Random initial states near the target and uniform random inputs within the actuator limits.

    ``scale`` shrinks both the initial states and the inputs, e.g. ``scale=0.02`` gives transitions close to the target for the
    local linear fit.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n_traj, horizon + 1, plant.n_x))
    u = rng.uniform(-plant.u_max, plant.u_max, size=(n_traj, horizon, plant.n_u)) * scale
    x[:, 0] = plant.sample_state(rng, n_traj) * scale
    for k in range(horizon):
        x[:, k + 1] = plant.step(x[:, k], u[:, k])
    return Trajectories(x=x, u=u)
