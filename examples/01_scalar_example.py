"""A scalar plant where the nominal Koopman certificate holds and the plant is unstable (Appendix B of the paper).

Plant:        x+ = 1.1 x + 0.2 x^2 + u            (unstable at the origin: A_f = 1.1, B_f = 1)
Embedding:    z = Phi(x) = [x, x^2]
The first row of the linear model below reproduces the plant exactly, but the second row (a regression fit away from the origin)
has a nonzero slope at the origin, which violates tangent invariance. Run it:  python examples/01_scalar_example.py
"""
import numpy as np

from tik import dlqr, spectral_radius

f = lambda x, u: 1.1 * x + 0.2 * x**2 + u
Phi = lambda x: np.array([x, x**2])
J = np.array([[1.0], [0.0]])                  # dPhi/dx at 0
A_f, B_f = np.array([[1.1]]), np.array([[1.0]])
Q, R = np.diag([1.0, 0.0]), np.eye(1)         # cost on the physical state x only

models = {
    "Standard (fit away from the origin)": (np.array([[1.1, 0.2], [-0.1, 0.9]]), np.array([[1.0], [-0.75]])),
    "TIK (A J = J A_f, B = J B_f)": (np.array([[1.1, 0.2], [0.0, 0.9]]), np.array([[1.0], [0.0]])),
}
for name, (A, B) in models.items():
    K, _ = dlqr(A, B, Q, R)
    A_cl = A - B @ K
    F = A_f - B_f @ K @ J                      # deployed Jacobian
    x_true, z_latent = 0.01, Phi(0.01)
    for _ in range(30):
        x_true = f(x_true, (-K @ Phi(x_true)).item())
        z_latent = A_cl @ z_latent
    rho = spectral_radius(A_cl)
    print(f"{name}")
    print(f"  K = {np.round(K, 3).ravel()}   rho(A_cl) = {rho:.3f} (nominal: {'stable' if rho < 1 else 'unstable'})")
    print(f"  F = {F.item():.3f} (plant: {'stable' if abs(F.item()) < 1 else 'unstable'})")
    print(f"  after 30 steps from x0 = 0.01: latent model predicts {z_latent[0]:.4f}, the plant is at {x_true:.4f}\n")
