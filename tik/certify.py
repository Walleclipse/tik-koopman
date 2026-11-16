"""Stability tests and the certified terminal set.

Naming follows the paper: ``A_cl = A - B K`` is the latent closed loop, ``F = A_f - B_f K J`` the deployed Jacobian on the plant
(Eq. 3), and ``E = J F - A_cl J`` the closed-loop tangent defect (Eq. 5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .fit import LocalFit, fit_error_bound
from .lqr import spectral_radius


def deployed_jacobian(A_f, B_f, K, J) -> np.ndarray:
    """``F = A_f - B_f K J``: the Jacobian of ``x -> f(x, -K Phi(x))`` at the target."""
    return A_f - B_f @ K @ J


def tangent_defect(A, B, A_f, B_f, K, J) -> np.ndarray:
    """``E = J F - A_cl J``. It is zero for every ``K`` when ``A J = J A_f`` and ``B = J B_f`` (tangent invariance)."""
    F = deployed_jacobian(A_f, B_f, K, J)
    return J @ F - (A - B @ K) @ J


def _smallest_singular_value_on_circle(M: np.ndarray, m: int, chunk: int = 4096) -> float:
    """``min_k sigma_min(s_k I - M)`` over ``m`` equally spaced points ``s_k`` of the unit circle, in chunks to bound memory."""
    eye = np.eye(len(M))
    best = np.inf
    for start in range(0, m, chunk):
        s = np.exp(2j * np.pi * np.arange(start, min(start + chunk, m)) / m)
        best = min(best, float(np.linalg.svd(s[:, None, None] * eye[None] - M[None], compute_uv=False)[:, -1].min()))
    return best


def distance_to_unit_circle(M: np.ndarray, n_points: int = 720, max_points: int = 1 << 18) -> float:
    """Conservative lower bound on ``d_1(M) = min_{|s|=1} sigma_min(sI - M)``.

    The minimum over ``n_points`` equally spaced points on the unit circle is corrected by the Lipschitz constant of
    ``sigma_min`` along the circle, ``2 sin(pi / (2 n))``. The grid is refined (never beyond ``max_points``) until the correction is
    below 10 % of the sampled minimum, so the bound stays informative for strongly nonnormal matrices. The SVDs run in chunks, so
    memory does not grow with the grid. It is a valid lower bound up to floating-point error in the SVDs.
    """
    m = min(n_points, max_points)
    while True:
        smin = _smallest_singular_value_on_circle(M, m)
        corr = 2 * np.sin(np.pi / (2 * m))
        if corr <= 0.1 * smin or m >= max_points:
            return max(0.0, smin - corr)
        m = min(4 * m, max_points)


def transfer_holds(defect: np.ndarray, A_cl: np.ndarray) -> bool:
    """Theorem 2(b): if ``A_cl`` is Schur and ``|E|_2 < d_1(A_cl)`` the plant is locally exponentially stable."""
    return spectral_radius(A_cl) < 1 and np.linalg.norm(defect, 2) < distance_to_unit_circle(A_cl)


@dataclass
class OfflineVerdict:
    verdict: str          # "stable", "unstable", or "inconclusive"
    rho_F_hat: float
    eta: float
    margin: float         # lower bound on the stability radius of F_hat

    def __str__(self) -> str:
        return f"{self.verdict} (rho(F_hat)={self.rho_F_hat:.4f}, eta={self.eta:.2e}, margin>={self.margin:.2e})"


def offline_stability_test(fit: LocalFit, K_o: np.ndarray, curvature: float, noise: float = 0.0,
                           third: float | None = None) -> OfflineVerdict:
    """Theorem 6: a stability verdict for the deployed loop from offline data only.

    ``K_o`` is the Jacobian of the deployed policy (``u = -K_o x`` locally); for TIK it equals the LQR gain of the fit.
    The test compares the fit error ``eta(Gamma)`` with a lower bound on the stability radius of
    ``F_hat = D_hat Gamma``,
    ``Gamma = [I; -K_o]``. We use ``d_1`` (a lower bound on the real stability radius), so the test can abstain more often than
    the exact one but never accepts when the exact one would not. The curvature and noise bounds are inputs; the verdict is only as
    valid as they are.
    """
    n_x = fit.A_f.shape[0]
    Gamma = np.vstack([np.eye(n_x), -K_o])
    F_hat = fit.D @ Gamma
    rho = spectral_radius(F_hat)
    eta = fit_error_bound(fit, Gamma, curvature, noise, third)
    margin = distance_to_unit_circle(F_hat)
    if eta < margin:
        return OfflineVerdict("stable" if rho < 1 else "unstable", rho, eta, margin)
    return OfflineVerdict("inconclusive", rho, eta, margin)


@dataclass
class TerminalSet:
    S: np.ndarray          # ellipsoid matrix, Omega_c = {x : x' S x <= c}
    c: float
    r: float               # Euclidean radius within which the contraction condition holds
    vartheta: float
    kappa: float

    def contains(self, x: np.ndarray) -> bool:
        return float(x @ self.S @ x) <= self.c


def _lyapunov_quantities(F_hat: np.ndarray, Q: np.ndarray):
    S = solve_discrete_lyapunov(F_hat.T, Q)
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    S_half = V @ np.diag(np.sqrt(w)) @ V.T
    S_half_inv = V @ np.diag(1 / np.sqrt(w)) @ V.T
    vartheta = float(np.linalg.norm(S_half @ F_hat @ S_half_inv, 2))
    kappa = float(w[-1] / w[0])
    return S, w, vartheta, kappa


def terminal_set(F_hat: np.ndarray, eta: float, L_H: float, L_T: float = 0.0, Q: np.ndarray | None = None) -> TerminalSet | None:
    """Theorem 8: a forward-invariant ellipsoid of ``x -> f(x, -K_hat x)``, or ``None`` if the condition cannot be met.

    Solves ``F_hat' S F_hat - S = -Q``, sets ``vartheta = |S^{1/2} F_hat S^{-1/2}|_2`` and ``kappa = cond(S)``, and takes the largest
    radius ``r`` with ``vartheta + sqrt(kappa) (eta + L_H r / 2 + L_T r^2 / 6) <= 1``. Any ``c <= lambda_min(S) r^2`` whose ellipsoid
    also fits the state and input limits is invariant. Intersecting with those limits is left to the caller. Returns ``None`` if
    ``F_hat`` is not Schur.

    Any ``Q > 0`` is allowed. If ``Q`` is None we search diagonal ``Q`` (log-uniform, fixed seed) for the one that maximizes the slack
    ``(1 - vartheta) / sqrt(kappa)``, which is the quantity ``eta`` has to stay below.
    """
    if spectral_radius(F_hat) >= 1:
        return None                     # no Lyapunov matrix S > 0 exists
    n = len(F_hat)
    if Q is None:
        rng = np.random.default_rng(0)
        best, Q = -np.inf, np.eye(n)
        for q in [np.ones(n)] + [10 ** rng.uniform(-3, 3, n) for _ in range(300)]:
            _, _, th, kap = _lyapunov_quantities(F_hat, np.diag(q))
            if (1 - th) / np.sqrt(kap) > best:
                best, Q = (1 - th) / np.sqrt(kap), np.diag(q)
    S, w, vartheta, kappa = _lyapunov_quantities(F_hat, Q)
    slack = (1 - vartheta) / np.sqrt(kappa) - eta
    if slack <= 0:
        return None
    if L_H == 0 and L_T == 0:
        r = np.inf
    elif L_T == 0:
        r = slack / (0.5 * L_H)
    else:
        a, b = L_T / 6.0, L_H / 2.0
        r = (-b + np.sqrt(b * b + 4 * a * slack)) / (2 * a)
    return TerminalSet(S=S, c=float(w[0] * r**2), r=float(r), vartheta=vartheta, kappa=kappa)


def estimate_remainder_constant(g, F: np.ndarray, radius: float, n_samples: int = 2000, safety: float = 2.0, seed: int = 0) -> float:
    """Empirical ``L_H`` with ``|g(x) - F x|_2 <= L_H |x|_2^2 / 2`` for ``|x|_2 <= radius`` (and ``L_T = 0``).

    ``g`` is the closed-loop map ``x -> f(x, -K_hat x)``. Sampled ratios times ``safety``: an estimate, not a proof.
    """
    rng = np.random.default_rng(seed)
    n = len(F)
    worst = 0.0
    for _ in range(n_samples):
        x = rng.normal(size=n)
        x *= radius * rng.uniform() ** (1.0 / n) / np.linalg.norm(x)
        worst = max(worst, 2 * float(np.linalg.norm(g(x) - F @ x)) / float(x @ x))
    return safety * worst
