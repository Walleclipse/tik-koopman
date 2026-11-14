"""Discrete-time LQR."""
from __future__ import annotations

import numpy as np


def solve_dare(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Stabilizing solution ``P`` of the discrete algebraic Riccati equation by the structured doubling algorithm.

    It needs ``(A, B)`` stabilizable and ``(A, Q^{1/2})`` detectable, and uses only matrix products and solves. We use it instead of
    ``scipy.linalg.solve_discrete_are`` because the QZ-based solver can stall on the ill-conditioned lifted models in some
    multi-threaded BLAS setups.
    """
    n = len(A)
    Ak, Gk, Hk = A.copy(), B @ np.linalg.solve(R, B.T), Q.copy()
    eye = np.eye(n)
    for _ in range(max_iter):
        W = np.linalg.solve(eye + Gk @ Hk, eye)
        A_next = Ak @ W @ Ak
        G_next = Gk + Ak @ W @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ W @ Ak
        done = np.max(np.abs(H_next - Hk)) <= tol * max(1.0, np.max(np.abs(H_next)))
        Ak, Gk, Hk = A_next, 0.5 * (G_next + G_next.T), 0.5 * (H_next + H_next.T)
        if done:
            return Hk
    raise RuntimeError("Riccati iteration did not converge: check stabilizability and detectability")


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gain ``K`` (``u = -K x``) and Riccati solution ``P`` of ``min sum x'Qx + u'Ru``."""
    P = solve_dare(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))
