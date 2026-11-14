"""Tangent-Invariant Koopman (TIK): Koopman control whose stability certificate holds on the plant."""
from .certify import (OfflineVerdict, TerminalSet, deployed_jacobian, distance_to_unit_circle, estimate_remainder_constant,
                      offline_stability_test, tangent_defect, terminal_set, transfer_holds)
from .control import KoopmanLQR, recovers, rollout
from .data import Trajectories, collect
from .fit import LocalFit, estimate_derivative_bound, fit_error_bound, fit_local_linearization
from .lqr import dlqr, spectral_radius
from .models import BaseKoopman, KoopmanModel, TIKKoopman
from .systems import CartPole
from .train import TrainConfig, finetune_tik, train_standard

__version__ = "0.1.0"
__all__ = [name for name in dir() if not name.startswith("_")]
