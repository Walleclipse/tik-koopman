"""Training: Standard Koopman, and TIK fine-tuning from a Standard Koopman checkpoint."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import Trajectories
from .fit import LocalFit
from .models import BaseKoopman, KoopmanModel, TIKKoopman


@dataclass
class TrainConfig:
    steps: int = 6000
    lr: float = 1e-3
    batch_size: int = 64
    horizon: int = 15
    gamma: float = 0.8
    control_loss_weight: float = 1.0
    rho_cap: float = 0.98
    rho_penalty: float = 100.0
    seed: int = 0
    log_every: int = 1000


def _windows(data: Trajectories, batch_size: int, horizon: int, rng: np.random.Generator):
    n, steps, _ = data.u.shape
    i = rng.integers(0, n, batch_size)
    k = rng.integers(0, steps - horizon + 1, batch_size)
    idx = k[:, None] + np.arange(horizon + 1)[None]
    x = torch.as_tensor(data.x[i[:, None], idx])
    u = torch.as_tensor(data.u[i[:, None], idx[:, :-1]])
    return x, u


def prediction_loss(model: BaseKoopman, x: torch.Tensor, u: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted multi-step state prediction error ``sum_k gamma^k |C z_{k+1} - x_{k+1}|^2``."""
    pred = model.predict(x[:, 0], u)
    w = gamma ** torch.arange(u.shape[1], dtype=pred.dtype)
    return ((pred - x[:, 1:]) ** 2).sum(-1).mul(w).sum(1).div(w.sum()).mean()


def inverse_control_loss(model: BaseKoopman, x: torch.Tensor, u: torch.Tensor, gamma: float) -> torch.Tensor:
    """Auxiliary loss ``|B^+ (Phi(x_{k+1}) - A Phi(x_k)) - u_k|^2`` that ties ``B u`` to the one-step latent transition."""
    A, B = model.matrices()
    z = model.encode(x)
    resid = (z[:, 1:] - z[:, :-1] @ A.T) @ torch.linalg.pinv(B).T
    w = gamma ** torch.arange(u.shape[1], dtype=resid.dtype)
    return ((resid - u) ** 2).mean(-1).mul(w).sum(1).div(w.sum()).mean()


def train_standard(model: KoopmanModel, data: Trajectories, cfg: TrainConfig = TrainConfig(), verbose: bool = True) -> KoopmanModel:
    """Standard Koopman: prediction loss plus the inverse control loss, all of ``(Phi, A, B)`` free."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.steps // 20), gamma=0.9)
    for it in range(1, cfg.steps + 1):
        x, u = _windows(data, cfg.batch_size, cfg.horizon, rng)
        loss = prediction_loss(model, x, u, cfg.gamma)
        if cfg.control_loss_weight:
            loss = loss + cfg.control_loss_weight * inverse_control_loss(model, x, u, cfg.gamma)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and it % cfg.log_every == 0:
            print(f"[standard] step {it:6d}  loss {loss.item():.5f}")
    return model


def finetune_tik(base: KoopmanModel, fit: LocalFit, data: Trajectories, cfg: TrainConfig = TrainConfig(),
                 bilinear: bool = False, verbose: bool = True) -> TIKKoopman:
    """Initialize TIK from a trained Standard Koopman model and fine-tune it under the tangent-invariant parameterization.

    The loss is the prediction loss plus a hinge penalty that keeps the spectral radius of the complement block below
    ``cfg.rho_cap`` (required by Proposition 4(b)).
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = TIKKoopman(base, fit.A_f, fit.B_f, bilinear=bilinear)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr * 0.3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    for it in range(1, cfg.steps + 1):
        x, u = _windows(data, cfg.batch_size, cfg.horizon, rng)
        loss = prediction_loss(model, x, u, cfg.gamma)
        rho = model.complement_radius()
        loss = loss + cfg.rho_penalty * torch.relu(rho - cfg.rho_cap) ** 2
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and it % cfg.log_every == 0:
            print(f"[tik]      step {it:6d}  loss {loss.item():.5f}  complement radius {rho.item():.3f}")
    return model
