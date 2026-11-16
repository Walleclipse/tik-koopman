"""Offline stability test (Theorem 6) and a certified terminal set (Theorem 8) for the cart-pole, from data only.

For transitions at several distances from the upright target:

1. fit the local linearization (Eq. 6) and its LQR gain K_hat (for TIK this is the deployed local gain, Prop. 4);
2. estimate curvature bounds (empirical, safety factor 2), bound the fit error eta (Eq. 8 with the cubic remainder of Corollary 12)
   and compare it with a lower bound on the stability margin of F_hat = A_f_hat - B_f_hat K_hat;
3. try to build the terminal ellipsoid of Theorem 8 and check by simulation that the local controller keeps sampled points inside.

The tests only certify when the fit is accurate enough: the Euclidean-norm terminal-set condition is the more demanding one.
Run:  python examples/03_offline_certificate.py
"""
import numpy as np

import tik

plant = tik.CartPole()
A_f, B_f = plant.linearize()                                    # simulator Jacobians, used only as a reference below
L_H = tik.estimate_derivative_bound(plant, order=2)                              # second derivative at the target

print(f"{'data scale':>10}  {'offline stability test':<64}  terminal set")
for scale in (1e-2, 1e-3, 1e-4, 1e-5):
    data = tik.collect(plant, n_traj=300, horizon=20, seed=1, scale=scale)       # transitions close to the target
    fit = tik.fit_local_linearization(*data.transitions(), radius=10.0)
    K_hat, _ = tik.dlqr(fit.A_f, fit.B_f, np.eye(4), np.eye(1))
    L_T = tik.estimate_derivative_bound(plant, order=3, radius=max(10 * scale, 0.05))
    verdict = tik.offline_stability_test(fit, K_hat, curvature=L_H, third=L_T)

    F_hat = fit.A_f - fit.B_f @ K_hat
    g = lambda x, K=K_hat: plant.step(x, -K @ x, clip=False)                      # local closed loop (inputs unsaturated near 0)
    L_closed = tik.estimate_remainder_constant(g, A_f - B_f @ K_hat, radius=0.05)
    omega = tik.terminal_set(F_hat, eta=verdict.eta, L_H=L_closed)
    if omega is None:
        status = "none (fit error too large for the contraction condition)"
    else:
        rng = np.random.default_rng(0)
        kept = 0
        for _ in range(2000):
            x = rng.normal(size=4)
            x *= np.sqrt(omega.c) / np.sqrt(x @ omega.S @ x) * rng.uniform() ** 0.25   # random point of the ellipsoid
            kept += omega.contains(g(x))
        status = f"c = {omega.c:.1e}; one step keeps {kept}/2000 sampled points inside"
    print(f"{scale:>10.0e}  {str(verdict):<64}  {status}")
print(f"\nreference (simulator Jacobians): rho(A_f - B_f K_hat) = {tik.spectral_radius(A_f - B_f @ K_hat):.4f}")
