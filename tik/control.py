"""Koopman LQR policy and closed-loop rollouts on the plant."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .lqr import dlqr, spectral_radius
from .models import BaseKoopman


@dataclass
class KoopmanLQR:
    """``u = -K Phi(x)`` with ``K`` the LQR gain of the lifted model (cost ``x'Q_x x + u'Ru`` on the physical state)."""

    model: BaseKoopman
    K: np.ndarray
    A: np.ndarray
    B: np.ndarray
    J: np.ndarray

    @classmethod
    def from_model(cls, model: BaseKoopman, Q_x: np.ndarray | None = None, R: np.ndarray | None = None) -> "KoopmanLQR":
        A, B = model.numpy_matrices()
        n_x, n_u, n_z = model.n_x, model.n_u, model.n_z
        Q = np.zeros((n_z, n_z))
        Q[:n_x, :n_x] = np.eye(n_x) if Q_x is None else Q_x
        K, _ = dlqr(A, B, Q, np.eye(n_u) if R is None else R)
        return cls(model=model, K=K, A=A, B=B, J=model.numpy_jacobian())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = self.model.encode(torch.as_tensor(x, dtype=torch.float64)).numpy()
        return -self.K @ z

    @property
    def latent_radius(self) -> float:
        """``rho(A - B K)``: the nominal certificate."""
        return spectral_radius(self.A - self.B @ self.K)

    def deployed_radius(self, A_f: np.ndarray, B_f: np.ndarray) -> float:
        """``rho(A_f - B_f K J)``: what the plant actually does near the target (``> 1`` means unstable)."""
        return spectral_radius(A_f - B_f @ self.K @ self.J)


def rollout(plant, policy, x0: np.ndarray, steps: int) -> np.ndarray:
    """Simulate ``x+ = f(x, policy(x))`` with the actuator limits of the plant. Returns ``(steps + 1, n_x)``."""
    xs = [np.asarray(x0, dtype=float)]
    for _ in range(steps):
        xs.append(plant.step(xs[-1], policy(xs[-1])))
        if not np.all(np.isfinite(xs[-1])) or np.linalg.norm(xs[-1]) > 1e3:
            break
    return np.array(xs)


def recovers(plant, policy, x0: np.ndarray, steps: int = 1500, tol: float = 1e-2) -> bool:
    """Does the closed loop return to within ``tol`` of the target in ``steps`` steps?"""
    traj = rollout(plant, policy, x0, steps)
    return len(traj) == steps + 1 and bool(np.linalg.norm(traj[-1]) < tol)
