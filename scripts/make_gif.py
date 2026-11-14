"""Render the README animation: Standard Koopman and TIK on the same cart-pole tilt, from models trained here.

Trains one Standard Koopman model and one TIK model exactly as ``examples/02_cartpole_standard_vs_tik.py`` does, simulates both local
feedback gains (``u = -K J x``) from the same initial tilt on the nonlinear plant, and writes a GIF.

    python scripts/make_gif.py --out docs/assets/cartpole.gif [--tilt 20]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import tik  # noqa: E402

COLORS = {"Standard Koopman": "#e8730c", "TIK": "#0a9d8f"}
POLE_LENGTH = 1.0            # drawn pole length (m); the plant's pole half-length is 0.5 m


def trained_gains(seed: int = 0) -> dict[str, np.ndarray]:
    """Deployed local gains ``K J`` of a Standard Koopman and a TIK model (same recipe as the cart-pole example)."""
    torch.set_num_threads(1)
    plant = tik.CartPole()
    data = tik.collect(plant, n_traj=1500, horizon=40, seed=0)
    near = tik.collect(plant, n_traj=300, horizon=20, seed=1, scale=0.02)
    fit = tik.fit_local_linearization(*near.transitions(), radius=1.0)
    torch.manual_seed(seed)
    quiet = dict(seed=seed, log_every=10**9)
    standard = tik.train_standard(tik.KoopmanModel(plant.n_x, plant.n_u), data, tik.TrainConfig(steps=20000, **quiet), verbose=False)
    tik_model = tik.finetune_tik(standard, fit, data, tik.TrainConfig(steps=5000, **quiet), verbose=False)
    gains = {}
    for name, model in (("Standard Koopman", standard), ("TIK", tik_model)):
        ctl = tik.KoopmanLQR.from_model(model)
        gains[name] = ctl.K @ ctl.J
    return gains


def simulate(plant, gain: np.ndarray, tilt_deg: float, steps: int) -> np.ndarray:
    """Plant rollout under ``u = -gain x``; frozen once the pole is horizontal."""
    x0 = np.array([0.0, np.deg2rad(tilt_deg), 0.0, 0.0])
    traj = tik.rollout(plant, lambda x: -gain @ x, x0, steps)
    fallen = np.flatnonzero(np.abs(traj[:, 1]) > np.pi / 2)
    if len(fallen):
        traj = np.vstack([traj[: fallen[0] + 1], np.repeat(traj[fallen[0]][None], steps + 1 - fallen[0] - 1, axis=0)])
    return traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cartpole.gif")
    ap.add_argument("--tilt", type=float, default=20.0, help="initial pole tilt in degrees")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--stride", type=int, default=2, help="simulation steps per frame")
    args = ap.parse_args()

    plant = tik.CartPole()
    steps = int(args.seconds / plant.dt)
    trajs = {name: simulate(plant, g, args.tilt, steps) for name, g in trained_gains().items()}

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), dpi=110)
    artists = {}
    for ax, (name, traj) in zip(axes, trajs.items()):
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-0.35, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, color=COLORS[name], fontweight="bold")
        ax.axhline(-0.12, color="#888888", lw=1.5)
        ax.add_patch(Rectangle((-0.2, -0.12), 0.4, 0.14, color="#1d1d1f"))
        pole, = ax.plot([], [], color=COLORS[name], lw=6, solid_capstyle="round")
        bob, = ax.plot([], [], "o", color="#1d1d1f", ms=9)
        label = ax.text(0, 1.12, "", ha="center", fontsize=11)
        artists[name] = (traj, pole, bob, label)

    def draw(frame):
        k = frame * args.stride
        for traj, pole, bob, label in artists.values():
            th = traj[k, 1]                                    # the camera follows the cart, so the cart stays at the origin
            tip = (POLE_LENGTH * np.sin(th), 0.02 + POLE_LENGTH * np.cos(th))
            pole.set_data([0, tip[0]], [0.02, tip[1]])
            bob.set_data([tip[0]], [tip[1]])
            label.set_text("falls" if abs(th) > np.pi / 2 else f"t = {k * plant.dt:.2f} s")
            label.set_color("#d93025" if abs(th) > np.pi / 2 else "#6e6e73")
        return [a for art in artists.values() for a in art[1:]]

    n_frames = steps // args.stride
    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=int(round(0.5 / (plant.dt * args.stride)))))   # half real-time speed
    print("saved", args.out)


if __name__ == "__main__":
    main()
