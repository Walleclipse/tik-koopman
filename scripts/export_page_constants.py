"""Print the small constants used by the project page (docs/index.html): local gains and spectra of one Standard Koopman and one TIK model.

The page animates the cart-pole in the browser. It needs the deployed local gain ``K J`` of each controller and the eigenvalues of the
latent closed loop ``A - B K`` and of the deployed Jacobian ``F = A_f - B_f K J``. This script trains the two models exactly as
``examples/02_cartpole_standard_vs_tik.py`` does (seed 0) and prints them as JSON.

    python scripts/export_page_constants.py
"""
import json

import numpy as np
import torch

import tik


def spectrum(M: np.ndarray) -> list[list[float]]:
    return [[float(z.real), float(z.imag)] for z in np.linalg.eigvals(M)]


def main(seed: int = 0, standard_steps: int = 20000, tik_steps: int = 5000) -> None:
    torch.set_num_threads(1)
    plant = tik.CartPole()
    A_f, B_f = plant.linearize()
    data = tik.collect(plant, n_traj=1500, horizon=40, seed=0)
    near = tik.collect(plant, n_traj=300, horizon=20, seed=1, scale=0.02)
    fit = tik.fit_local_linearization(*near.transitions(), radius=1.0)

    torch.manual_seed(seed)
    standard = tik.train_standard(tik.KoopmanModel(plant.n_x, plant.n_u), data,
                                  tik.TrainConfig(steps=standard_steps, seed=seed, log_every=10**9), verbose=False)
    tik_model = tik.finetune_tik(standard, fit, data, tik.TrainConfig(steps=tik_steps, seed=seed, log_every=10**9), verbose=False)

    out = {}
    for name, model in (("standard", standard), ("tik", tik_model)):
        ctl = tik.KoopmanLQR.from_model(model)
        out[name] = {
            "local_gain": (ctl.K @ ctl.J).ravel().round(5).tolist(),
            "latent_spectrum": spectrum(ctl.A - ctl.B @ ctl.K),
            "plant_spectrum": spectrum(A_f - B_f @ ctl.K @ ctl.J),
        }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
