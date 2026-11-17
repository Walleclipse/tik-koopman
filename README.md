# Making Koopman Control Guarantees Hold on Robots

**Tangent-Invariant Koopman (TIK): certified on the model, stable on the robot.**

[![Project page](https://img.shields.io/badge/Project%20page-online-0f766e)](https://walleclipse.github.io/tik-koopman/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.20402-b31b1b.svg)](https://arxiv.org/abs/2611.20402)

Abulikemu Abuduweili and Changliu Liu · Robotics Institute, Carnegie Mellon University

<p align="center">
  <img src="docs/assets/cartpole.gif" width="720" alt="Cart-pole from a 20 degree tilt: Standard Koopman falls, TIK balances">
</p>

*Same cart-pole, same data, same 20° tilt. Both controllers pass the stability test on their learned model. Standard Koopman falls. TIK balances.*

## Motivation

A Koopman model describes a nonlinear robot with a linear model in a learned latent space. Control on a linear model is fast, and the controller comes with a stability test.

The test checks the model's own loop. On the robot, every new measurement is encoded again, which gives a different loop. A controller that passes the test can still fall.

## Method

TIK makes the model agree with the robot's linearization at the target. Then the stability test also holds on the robot.

1. **Fit** the robot's linearization `Â_f, B̂_f` from offline data near the target.
2. **Build** the model matrices from the fit and the encoder Jacobian `J`:
   ```
   A = J Â_f J⁺ + W P⊥        B = J B̂_f
   ```
   `P⊥` keeps the directions that `J` cannot reach, and `W` is free and learned.
3. **Control and check.** Use LQR or MPC on the model, test its stability from the same data, and plan large moves with MPPI.

## Results

All experiments are in simulation. Models that are stable on the robot, out of 10:

| Controller | Cart-pole (ODE) | Cart-pole (MuJoCo) | Double cart-pole | Double pendulum |
|---|:---:|:---:|:---:|:---:|
| Standard Koopman | 0 | 0 | 0 | 0 |
| Koopman + invariance penalty | 10 | 10 | 0 | 7 |
| Deep Koopman with control | 10 | 10 | 0 | 9 |
| Neural model + NMPC | 10 | 10 | 1 | 0 |
| **TIK (ours)** | **10** | **10** | **10** | **10** |

- Swing-up succeeds in 100% of runs at 2 ms per planning step and 88% at 0.25 ms, where the neural planner cannot run.
- On a 7-joint Kinova arm, the MPC problem is solved in 2.1 ms (median). Linear MPC reaches the same success rate, so this task tests speed.
- The offline stability test, which uses estimated curvature bounds, passes for 20 of 20 TIK controllers at low noise. It is inconclusive at high noise.

The paper has the full tables, including the Unitree Go2 and the six-goal arm.

## Installation

```bash
git clone https://github.com/Walleclipse/tik-koopman.git
cd tik-koopman
pip install -e ".[plots,dev]"
pytest -q
```

## Quickstart

```python
import torch
import tik

plant = tik.CartPole()
A_f, B_f = plant.linearize()          # true linearization, used only to check the result

# Offline data: random trajectories, and transitions close to the target for the local fit
data = tik.collect(plant, n_traj=1500, horizon=40)
near = tik.collect(plant, n_traj=300, horizon=20, scale=0.02)
fit = tik.fit_local_linearization(*near.transitions(), radius=1.0)

# Standard Koopman, then TIK fine-tuned from it
torch.manual_seed(0)
standard = tik.train_standard(tik.KoopmanModel(plant.n_x, plant.n_u), data, tik.TrainConfig(steps=20000), verbose=False)
model = tik.finetune_tik(standard, fit, data, tik.TrainConfig(steps=5000), verbose=False)

# rho < 1 means stable. "model" is what the test checks, "robot" is what the robot does.
for name, m in [("Standard Koopman", standard), ("TIK", model)]:
    ctl = tik.KoopmanLQR.from_model(m)
    print(f"{name:17s} model: {ctl.latent_radius:.3f}   robot: {ctl.deployed_radius(A_f, B_f):.3f}")
```

Standard Koopman passes on the model and fails on the robot. For TIK, both numbers agree and are below 1.

## Examples

| File | What it shows |
|---|---|
| [`01_scalar_example.py`](examples/01_scalar_example.py) | A one-dimensional toy: an accurate model and an unstable robot. |
| [`02_cartpole_standard_vs_tik.py`](examples/02_cartpole_standard_vs_tik.py) | Standard Koopman and TIK on the cart-pole, over several seeds. |
| [`03_offline_certificate.py`](examples/03_offline_certificate.py) | A stability test and a terminal set from offline data. |

## Citation

```bibtex
@article{abuduweili2026tik,
  title   = {Making Koopman Control Guarantees Hold on Robots},
  author  = {Abuduweili, Abulikemu and Liu, Changliu},
  journal = {arXiv preprint arXiv:2611.20402},
  year    = {2026},
  url={https://arxiv.org/abs/2611.20402}, 
}
```
