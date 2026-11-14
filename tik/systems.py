"""Plants used in the examples and tests.

A plant is a discrete-time map ``x+ = f(x, u)`` with a target equilibrium at the origin. Everything in ``tik`` only needs ``step``,
``n_x``, ``n_u`` and the input limits, so adding a plant is a few lines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartPole:
    """Cart-pole with the pole upright at the target, state ``(x, theta, x_dot, theta_dot)``.

    The upright equilibrium is unstable without control. ``theta`` is measured from upright. The continuous dynamics are
    integrated with ``substeps`` RK4 steps per control period ``dt``.
    """

    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_half_length: float = 0.5
    dt: float = 0.02
    substeps: int = 4
    force_limit: float = 10.0
    n_x: int = 4
    n_u: int = 1

    @property
    def u_max(self) -> np.ndarray:
        return np.array([self.force_limit])

    def _deriv(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        _, th, xd, thd = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        mc, mp, l, g = self.cart_mass, self.pole_mass, self.pole_half_length, self.gravity
        total = mc + mp
        c, s = np.cos(th), np.sin(th)
        tmp = (u[..., 0] + mp * l * thd**2 * s) / total
        thacc = (g * s - c * tmp) / (l * (4.0 / 3.0 - mp * c**2 / total))
        xacc = tmp - mp * l * thacc * c / total
        return np.stack([xd, thd, xacc, thacc], axis=-1)

    def step(self, x: np.ndarray, u: np.ndarray, clip: bool = True) -> np.ndarray:
        """One control period. Accepts a single state ``(n_x,)`` or a batch ``(..., n_x)``."""
        x = np.asarray(x, dtype=float)
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if clip:
            u = np.clip(u, -self.force_limit, self.force_limit)
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = self._deriv(x, u)
            k2 = self._deriv(x + 0.5 * h * k1, u)
            k3 = self._deriv(x + 0.5 * h * k2, u)
            k4 = self._deriv(x + h * k3, u)
            x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def linearize(self, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Jacobians ``(A_f, B_f)`` of ``step`` at the origin by central differences (the *oracle* linearization)."""
        x0, u0 = np.zeros(self.n_x), np.zeros(self.n_u)
        A = np.stack([(self.step(x0 + eps * e, u0, clip=False) - self.step(x0 - eps * e, u0, clip=False)) / (2 * eps)
                      for e in np.eye(self.n_x)], axis=1)
        B = np.stack([(self.step(x0, u0 + eps * e, clip=False) - self.step(x0, u0 - eps * e, clip=False)) / (2 * eps)
                      for e in np.eye(self.n_u)], axis=1)
        return A, B

    def sample_state(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Random states near the upright, where the local controller operates."""
        lo = np.array([-0.5, -0.2, -0.5, -0.5])
        return rng.uniform(lo, -lo, size=(n, self.n_x))
