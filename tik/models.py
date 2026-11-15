"""Neural Koopman models: Standard Koopman (free linear predictor) and Tangent-Invariant Koopman (TIK).

Notation follows the paper. The embedding is ``z = Phi(x) = [x; Psi(x) - Psi(0)]`` with an MLP ``Psi``, so ``Phi(0) = 0`` and the
state is read out as ``x = C z`` with ``C = [I 0]``. The linear predictor is ``z+ = A z + B u``. ``J`` is the Jacobian of ``Phi`` at
the target and ``J^+`` its pseudoinverse. All tensors are float64.
"""
from __future__ import annotations

import torch
from torch import nn


class BaseKoopman(nn.Module):
    """Encoder, rollout and Jacobian shared by both models. Subclasses define ``matrices()`` -> ``(A, B)``."""

    def __init__(self, n_x: int, n_u: int, latent_per_state: int = 8, width: int = 128, depth: int = 2):
        super().__init__()
        self.n_x, self.n_u = n_x, n_u
        self.n_z = n_x * latent_per_state
        self.width, self.depth = width, depth
        layers: list[nn.Module] = []
        d = n_x
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, self.n_z - n_x))
        self.psi = nn.Sequential(*layers)

    # -- embedding --------------------------------------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``Phi(x) = [x; Psi(x) - Psi(0)]`` for a state ``(n_x,)`` or a batch ``(..., n_x)``."""
        origin = torch.zeros(self.n_x, dtype=x.dtype, device=x.device)
        return torch.cat([x, self.psi(x) - self.psi(origin)], dim=-1)

    def jacobian(self, create_graph: bool = False) -> torch.Tensor:
        """``J = dPhi/dx`` at the target, shape ``(n_z, n_x)``. Differentiable when ``create_graph`` is set."""
        p = next(self.parameters())
        origin = torch.zeros(self.n_x, dtype=p.dtype, device=p.device)
        return torch.autograd.functional.jacobian(self.encode, origin, create_graph=create_graph)

    # -- dynamics ---------------------------------------------------------------------------------------------------
    def matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def step(self, z: torch.Tensor, u: torch.Tensor, matrices: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        A, B = matrices
        return z @ A.T + u @ B.T

    def predict(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Open-loop latent rollout from ``Phi(x0)``. ``u`` is ``(batch, H, n_u)``. Returns states ``(batch, H, n_x)``."""
        matrices = self.matrices()
        z = self.encode(x0)
        states = []
        for k in range(u.shape[1]):
            z = self.step(z, u[:, k], matrices)
            states.append(z[..., : self.n_x])
        return torch.stack(states, dim=1)

    # -- numpy views used by the controllers and the certificates ---------------------------------------------------
    @torch.no_grad()
    def numpy_matrices(self):
        A, B = self.matrices()
        return A.numpy().copy(), B.numpy().copy()

    @torch.no_grad()
    def numpy_jacobian(self):
        return self.jacobian().numpy().copy()


class KoopmanModel(BaseKoopman):
    """Standard Koopman: ``A`` and ``B`` are free parameters trained jointly with the encoder."""

    def __init__(self, n_x: int, n_u: int, latent_per_state: int = 8, width: int = 128, depth: int = 2):
        super().__init__(n_x, n_u, latent_per_state, width, depth)
        self.A = nn.Parameter(torch.eye(self.n_z) + 0.01 * torch.randn(self.n_z, self.n_z))
        self.B = nn.Parameter(0.01 * torch.randn(self.n_z, n_u))
        self.double()

    def matrices(self):
        return self.A, self.B


class TIKKoopman(BaseKoopman):
    """Tangent-Invariant Koopman. ``A`` and ``B`` are rebuilt from the encoder Jacobian ``J`` at every call::

        A = J A_f_hat J^+ + W (I - J J^+),     B = J B_f_hat

    so ``A J = J A_f_hat`` and ``B = J B_f_hat`` hold exactly for every ``W`` and every encoder weight. ``(A_f_hat, B_f_hat)`` is a
    local linearization of the plant fitted from offline data (``tik.fit``); ``W`` only acts on the complement of the tangent space.
    The encoder and ``W`` are initialized from a trained Standard Koopman model (``base``).

    ``bilinear=True`` adds ``sum_j u_j N_j z`` to the predictor. The term and its first derivatives vanish at the centered target, so
    the local guarantees are unchanged while the input effect can vary with the state.
    """

    def __init__(self, base: KoopmanModel, A_f_hat, B_f_hat, bilinear: bool = False):
        super().__init__(base.n_x, base.n_u, latent_per_state=base.n_z // base.n_x, width=base.width, depth=base.depth)
        self.psi.load_state_dict(base.psi.state_dict())
        self.W = nn.Parameter(base.A.detach().clone())
        self.N = nn.Parameter(torch.zeros(base.n_u, self.n_z, self.n_z)) if bilinear else None
        self.register_buffer("A_f_hat", torch.as_tensor(A_f_hat, dtype=torch.float64))
        self.register_buffer("B_f_hat", torch.as_tensor(B_f_hat, dtype=torch.float64))
        self.double()

    def matrices(self):
        J = self.jacobian(create_graph=True)
        J_pinv = torch.linalg.pinv(J)
        complement = torch.eye(self.n_z, dtype=J.dtype, device=J.device) - J @ J_pinv
        return J @ self.A_f_hat @ J_pinv + self.W @ complement, J @ self.B_f_hat

    def step(self, z, u, matrices):
        z_next = super().step(z, u, matrices)
        if self.N is not None:
            z_next = z_next + torch.einsum("...j,jkl,...l->...k", u, self.N, z)
        return z_next

    def complement_radius(self) -> torch.Tensor:
        """Spectral radius of ``(I - J J^+) W (I - J J^+)``. The LQR guarantee (Proposition 4) needs it below one."""
        J = self.jacobian(create_graph=True)
        complement = torch.eye(self.n_z, dtype=J.dtype, device=J.device) - J @ torch.linalg.pinv(J)
        return torch.linalg.eigvals(complement @ self.W @ complement).abs().max()
