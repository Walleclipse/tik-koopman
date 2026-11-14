"""Standard Koopman vs TIK on the cart-pole (the main phenomenon of the paper), end to end on a laptop CPU.

For every seed we train a Standard Koopman model, fine-tune a TIK model from it, and compare

  rho(A_cl)  the nominal certificate on the learned latent loop (< 1 passes),
  rho(F)     the spectral radius of the deployed Jacobian on the plant, F = A_f - B_f K J (< 1 means the plant is stable),

where (A_f, B_f) are the simulator's Jacobians (used only for evaluation). A third check simulates the plant from a
0.5 rad tilt, with the lifted controller u = -K Phi(x) and with its local gain u = -K J x.

  python examples/02_cartpole_standard_vs_tik.py [--seeds 3] [--standard-steps 20000] [--tik-steps 5000] [--plot figure.png]

The paper trains for 60,000 steps with 10 seeds per system. With fewer steps, some Standard Koopman seeds do not even pass the
nominal test (rho(A_cl) >= 1); the point is that those that do still fail on the plant, and that TIK passes both.
Set OMP_NUM_THREADS=1 if your BLAS setup is unstable.
"""
import argparse

import numpy as np
import torch

import tik

TILT = np.array([0.0, 0.5, 0.0, 0.0])        # 0.5 rad (29 degrees) from upright


def train_pair(plant, data, fit, seed, standard_steps, tik_steps):
    """Train a Standard Koopman model and a TIK model fine-tuned from it."""
    torch.manual_seed(seed)
    quiet = dict(seed=seed, log_every=10**9)
    standard = tik.train_standard(tik.KoopmanModel(plant.n_x, plant.n_u), data,
                                  tik.TrainConfig(steps=standard_steps, **quiet), verbose=False)
    tik_model = tik.finetune_tik(standard, fit, data, tik.TrainConfig(steps=tik_steps, **quiet), verbose=False)
    return {"standard": standard, "TIK": tik_model}


def report(plant, name, seed, controller, A_f, B_f):
    rho_cl, rho_F = controller.latent_radius, controller.deployed_radius(A_f, B_f)
    local_gain = controller.K @ controller.J
    lifted_ok = tik.recovers(plant, controller, TILT)
    local_ok = tik.recovers(plant, lambda x: -local_gain @ x, TILT)
    print(f"{seed:>4}  {name:<9}  {rho_cl:>9.4f}  {'passes' if rho_cl < 1 else 'fails':>8}  {rho_F:>10.4f}  "
          f"{'stable' if rho_F < 1 else 'UNSTABLE':>8}  {lifted_ok!s:>5} / {local_ok}")


def plot_first_seed(plant, controllers, path):
    """Pole angle after a 0.6 degree tilt: plant rollout (solid) against the latent model's own prediction (dashed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0 = np.array([0.0, np.deg2rad(0.6), 0.0, 0.0])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4), sharey=True)
    for ax, (name, ctl) in zip(axes, controllers.items()):
        trace = tik.rollout(plant, ctl, x0, 250)
        fallen = np.flatnonzero(np.abs(trace[:, 1]) > 0.5)
        trace = trace[: fallen[0] + 1] if len(fallen) else trace              # stop the trace when the pole falls
        z = ctl.model.encode(torch.as_tensor(x0)).detach().numpy()
        latent = []
        for _ in range(250):
            latent.append(z[1])
            z = (ctl.A - ctl.B @ ctl.K) @ z
        ax.plot(np.arange(len(trace)) * plant.dt, np.degrees(trace[:, 1]), label="plant")
        ax.plot(np.arange(250) * plant.dt, np.degrees(latent), "--", label="latent model")
        ax.set(title=name, xlabel="time (s)", ylim=(-30, 30))
    axes[0].set_ylabel("pole angle (deg)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("saved", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--standard-steps", type=int, default=20000)
    ap.add_argument("--tik-steps", type=int, default=5000)
    ap.add_argument("--plot", type=str, default="", help="save a figure of the first seed (needs matplotlib)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    plant = tik.CartPole()
    A_f, B_f = plant.linearize()
    data = tik.collect(plant, n_traj=1500, horizon=40, seed=0)                          # offline data over the state space
    near = tik.collect(plant, n_traj=300, horizon=20, seed=1, scale=0.02)               # transitions near the target
    fit = tik.fit_local_linearization(*near.transitions(), radius=1.0)
    K_hat, _ = tik.dlqr(fit.A_f, fit.B_f, np.eye(4), np.eye(1))

    header = f"{'seed':>4}  {'model':<9}  {'rho(A_cl)':>9}  {'nominal':>8}  {'rho(F)':>10}  {'plant':>8}"
    print(header + "  recovers from 0.5 rad: lifted / local gain")
    first = None
    for seed in range(args.seeds):
        models = train_pair(plant, data, fit, seed, args.standard_steps, args.tik_steps)
        controllers = {name: tik.KoopmanLQR.from_model(m) for name, m in models.items()}
        for name, ctl in controllers.items():
            report(plant, name, seed, ctl, A_f, B_f)
        tik_ctl = controllers["TIK"]
        print(f"      TIK local gain check: max|K J - K_hat| = {np.abs(tik_ctl.K @ tik_ctl.J - K_hat).max():.1e} (Proposition 4: zero)")
        first = first or controllers
    if args.plot:
        plot_first_seed(plant, first, args.plot)


if __name__ == "__main__":
    main()
