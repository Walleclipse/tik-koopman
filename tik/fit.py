"""Local linearization of the plant from offline transitions (Eq. 6) and its error bound (Eq. 8)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalFit:
    """Weighted least-squares fit ``x+ ~ D p + c`` with ``p = [x; u]`` and ``D = [A_f_hat  B_f_hat]``."""

    A_f: np.ndarray
    B_f: np.ndarray
    c: np.ndarray
    p: np.ndarray        # regressors, (M, n_x + n_u)
    w: np.ndarray        # weights, (M,)
    Sigma_inv: np.ndarray  # inverse of sum_i w_i q_i q_i', q_i = [p_i; 1]

    @property
    def D(self) -> np.ndarray:
        return np.hstack([self.A_f, self.B_f])


def fit_local_linearization(x: np.ndarray, u: np.ndarray, x_next: np.ndarray, radius: float | None = None,
                            bandwidth: float | None = None) -> LocalFit:
    """Fit the plant's first-order model near the target (the origin).

    Transitions farther than ``radius`` from the target (in ``|[x; u]|_2``) are dropped. Weights decrease with that distance as
    ``exp(-(d / bandwidth)^2)``; by default the bandwidth is the median distance of the kept transitions.
    """
    p = np.hstack([x, u])
    d = np.linalg.norm(p, axis=1)
    keep = d <= (radius if radius is not None else np.quantile(d, 0.2))
    p, y, d = p[keep], x_next[keep], d[keep]
    bw = bandwidth if bandwidth is not None else float(np.median(d))
    w = np.exp(-((d / bw) ** 2))
    q = np.hstack([p, np.ones((len(p), 1))])
    Sigma = (q * w[:, None]).T @ q
    if np.linalg.matrix_rank(Sigma) < Sigma.shape[0]:
        raise ValueError("weighted design is rank deficient; use more transitions or a larger radius")
    Sigma_inv = np.linalg.inv(Sigma)
    H = (y * w[:, None]).T @ q @ Sigma_inv           # (n_x, n_x + n_u + 1)
    n_x = x.shape[1]
    return LocalFit(A_f=H[:, :n_x], B_f=H[:, n_x:-1], c=H[:, -1], p=p, w=w, Sigma_inv=Sigma_inv)


def fit_error_bound(fit: LocalFit, Gamma: np.ndarray, curvature: float, noise: float = 0.0, third: float | None = None) -> float:
    """``eta(Gamma)`` in Eq. (8): a bound on ``|(D_hat - D_f) Gamma|_2``.

    Needs ``curvature``, a bound ``L`` on the Hessian of the plant on the fit neighborhood, and ``noise``, a bound on the
    successor measurement error. Both are assumptions (Assumption 1); estimate ``L`` conservatively.
    Each transition contributes ``w_i rho_i |Gamma' g_i|_2`` with ``rho_i = L/2 |p_i|^2 + eps`` and ``g_i`` the first
    ``n_x + n_u`` entries of ``Sigma^{-1} [p_i; 1]``. If ``third`` (a bound on the third derivative on the neighborhood) is given,
    ``curvature`` is read as the second derivative *at the target* and ``rho_i = L_H/2 |p_i|^2 + L_T/6 |p_i|^3 + eps`` (Corollary 4).
    That is much tighter at equilibria where the second derivative vanishes, as for an upright pendulum.
    """
    q = np.hstack([fit.p, np.ones((len(fit.p), 1))])
    g = (q @ fit.Sigma_inv)[:, : fit.p.shape[1]]
    r2 = np.sum(fit.p**2, axis=1)
    rho = 0.5 * curvature * r2 + noise + (0.0 if third is None else third / 6.0 * r2**1.5)
    return float(np.sum(fit.w * rho * np.linalg.norm(g @ Gamma, axis=1)))


def estimate_derivative_bound(plant, order: int, radius: float = 0.0, n_samples: int = 2000, safety: float = 2.0,
                              step: float = 1e-2, seed: int = 0) -> float:
    """Empirical bound on the ``order``-th derivative of ``p = [x; u] -> plant.step(x, u)`` (``order`` is 2 or 3).

    The directional derivative ``|D^k f(p)[a_1, ..., a_k]|`` is estimated by a finite difference along random unit directions
    ``a_i``, at random points ``p`` with ``|p|_2 <= radius`` (``radius = 0`` evaluates at the target only), maximized over the
    samples and multiplied by ``safety``. Use ``order=2, radius=0`` for ``L_H`` and ``order=3`` for ``L_T`` of the cubic remainder, or
    ``order=2, radius>0`` for the single constant ``L`` of Assumption 1. These are *estimates*: the paper's guarantees need valid
    bounds, so verdicts that use them are empirical.
    """
    rng = np.random.default_rng(seed)
    n = plant.n_x + plant.n_u

    def f(p):
        return plant.step(p[: plant.n_x], p[plant.n_x:], clip=False)

    worst = 0.0
    for _ in range(n_samples):
        p = rng.normal(size=n)
        p *= radius * rng.uniform() ** (1.0 / n) / np.linalg.norm(p)
        directions = rng.normal(size=(order, n))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        # k-th order finite difference: sum over subsets S of (-1)^(k - |S|) f(p + h sum_{i in S} a_i), divided by h^k
        total = 0.0
        for mask in range(2**order):
            subset = [i for i in range(order) if mask >> i & 1]
            total = total + (-1) ** (order - len(subset)) * f(p + step * directions[subset].sum(axis=0))
        worst = max(worst, float(np.linalg.norm(total)) / step**order)
    return safety * worst
