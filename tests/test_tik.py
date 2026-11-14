"""Unit tests for the identities the paper proves. Run: pytest -q"""
import numpy as np
import pytest
import torch

import tik
from tik.lqr import solve_dare

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def plant():
    return tik.CartPole()


@pytest.fixture(scope="module")
def tik_model(plant):
    """A TIK model on top of a random (untrained) encoder with a random W: the identities hold for every encoder and every W."""
    torch.manual_seed(0)
    base = tik.KoopmanModel(plant.n_x, plant.n_u, latent_per_state=4)
    A_f, B_f = plant.linearize()
    model = tik.TIKKoopman(base, A_f, B_f)
    with torch.no_grad():
        model.W.copy_(0.3 * torch.randn_like(model.W) / np.sqrt(model.n_z))      # small W keeps the complement block Schur
    return model, A_f, B_f


def test_plant_is_unstable_at_the_target(plant):
    A_f, _ = plant.linearize()
    assert tik.spectral_radius(A_f) > 1


def test_tangent_invariance_holds_exactly(tik_model):
    model, A_f, B_f = tik_model
    A, B = model.numpy_matrices()
    J = model.numpy_jacobian()
    assert np.allclose(A @ J, J @ A_f, atol=1e-10)             # A J = J A_f_hat
    assert np.allclose(B, J @ B_f, atol=1e-10)                 # B = J B_f_hat


def test_local_gain_equals_lqr_of_the_fit(tik_model):
    """Proposition 4(b): K J equals the LQR gain of (A_f_hat, B_f_hat) when the complement block is Schur."""
    model, A_f, B_f = tik_model
    assert model.complement_radius().item() < 1
    ctl = tik.KoopmanLQR.from_model(model)
    K_hat, _ = tik.dlqr(A_f, B_f, np.eye(4), np.eye(1))
    assert np.allclose(ctl.K @ ctl.J, K_hat, atol=1e-7)
    assert np.isclose(ctl.deployed_radius(A_f, B_f), tik.spectral_radius(A_f - B_f @ K_hat))


def test_certificate_transfers_under_tangent_invariance(tik_model):
    """Theorem 2(a): E = 0, so every eigenvalue of F is an eigenvalue of A_cl."""
    model, A_f, B_f = tik_model
    ctl = tik.KoopmanLQR.from_model(model)
    E = tik.tangent_defect(ctl.A, ctl.B, A_f, B_f, ctl.K, ctl.J)
    assert np.linalg.norm(E) < 1e-8
    eig_F = np.linalg.eigvals(tik.deployed_jacobian(A_f, B_f, ctl.K, ctl.J))
    eig_cl = np.linalg.eigvals(ctl.A - ctl.B @ ctl.K)
    assert all(np.min(np.abs(eig_cl - lam)) < 1e-6 for lam in eig_F)
    assert ctl.latent_radius >= tik.spectral_radius(tik.deployed_jacobian(A_f, B_f, ctl.K, ctl.J)) - 1e-9


def test_free_model_violates_tangent_invariance(plant):
    """A randomly initialized free model does not match the plant on the tangent space."""
    torch.manual_seed(1)
    model = tik.KoopmanModel(plant.n_x, plant.n_u, latent_per_state=4)
    A_f, B_f = plant.linearize()
    A, B = model.numpy_matrices()
    J = model.numpy_jacobian()
    assert np.linalg.norm(A @ J - J @ A_f) > 0.1


def test_scalar_example_matches_the_paper():
    A, B = np.array([[1.1, 0.2], [-0.1, 0.9]]), np.array([[1.0], [-0.75]])
    K, _ = tik.dlqr(A, B, np.diag([1.0, 0.0]), np.eye(1))
    J = np.array([[1.0], [0.0]])
    assert np.allclose(K, [[0.022, -0.863]], atol=1e-3)
    assert np.isclose(tik.spectral_radius(A - B @ K), 0.951, atol=1e-3)
    assert np.isclose((1.1 - K @ J).item(), 1.078, atol=1e-3)


def test_doubling_riccati_solver_matches_scipy(plant):
    from scipy.linalg import solve_discrete_are
    A, B = plant.linearize()
    Q, R = np.eye(4), np.eye(1)
    assert np.allclose(solve_dare(A, B, Q, R), solve_discrete_are(A, B, Q, R), rtol=1e-7, atol=1e-7)


def test_unit_circle_bound_is_a_lower_bound():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5))
    M *= 0.9 / tik.spectral_radius(M)
    s = np.exp(2j * np.pi * np.arange(100000) / 100000)
    exact = min(np.linalg.svd(si * np.eye(5) - M, compute_uv=False)[-1] for si in s[::50])
    lower = tik.distance_to_unit_circle(M)
    assert 0 < lower <= exact + 1e-9


def test_fit_error_bound_covers_the_true_error(plant):
    A_f, B_f = plant.linearize()
    data = tik.collect(plant, 200, 20, seed=3, scale=1e-3)
    fit = tik.fit_local_linearization(*data.transitions(), radius=10.0)
    K, _ = tik.dlqr(A_f, B_f, np.eye(4), np.eye(1))
    Gamma = np.vstack([np.eye(4), -K])
    err = np.linalg.norm((fit.D - np.hstack([A_f, B_f])) @ Gamma, 2)
    eta = tik.fit_error_bound(fit, Gamma, curvature=tik.estimate_derivative_bound(plant, order=2),
                              third=tik.estimate_derivative_bound(plant, order=3, radius=0.05, n_samples=500))
    assert err <= eta


def test_offline_test_is_stable_for_the_lqr_gain_of_a_good_fit(plant):
    data = tik.collect(plant, 300, 20, seed=1, scale=1e-3)
    fit = tik.fit_local_linearization(*data.transitions(), radius=10.0)
    K, _ = tik.dlqr(fit.A_f, fit.B_f, np.eye(4), np.eye(1))
    verdict = tik.offline_stability_test(fit, K, curvature=tik.estimate_derivative_bound(plant, order=2),
                                         third=tik.estimate_derivative_bound(plant, order=3, radius=0.05, n_samples=500))
    assert verdict.verdict == "stable"


def test_terminal_set_is_invariant_on_samples(plant):
    A_f, B_f = plant.linearize()
    K, _ = tik.dlqr(A_f, B_f, np.eye(4), np.eye(1))
    F = A_f - B_f @ K
    g = lambda x: plant.step(x, -K @ x, clip=False)
    L_H = tik.estimate_remainder_constant(g, F, radius=0.05, n_samples=500)
    omega = tik.terminal_set(F, eta=1e-5, L_H=L_H)
    assert omega is not None
    rng = np.random.default_rng(0)
    for _ in range(300):
        x = rng.normal(size=4)
        x *= np.sqrt(omega.c) / np.sqrt(x @ omega.S @ x) * rng.uniform() ** 0.25
        assert omega.contains(g(x))


def test_training_reduces_prediction_loss(plant):
    torch.manual_seed(0)
    data = tik.collect(plant, 200, 30, seed=0)
    model = tik.KoopmanModel(plant.n_x, plant.n_u, latent_per_state=4, width=64)
    rng = np.random.default_rng(0)
    from tik.train import _windows, prediction_loss
    x, u = _windows(data, 64, 10, rng)
    before = prediction_loss(model, x, u, 0.8).item()
    tik.train_standard(model, data, tik.TrainConfig(steps=300, horizon=10, log_every=10**9, control_loss_weight=0.0), verbose=False)
    assert prediction_loss(model, x, u, 0.8).item() < before
