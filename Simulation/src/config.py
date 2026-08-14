
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

@dataclass
class PriorConfig:
    mean_neural_sd: float = 0.35
    alpha_neural_sd: float = 0.50
    range_neural_sd: float = 0.35
    nu_shape: float = 2.0
    nu_rate: float = 0.5
    log_phi_mean: float = -1.0
    log_phi_sd: float = 0.75

@dataclass
class SimulationConfig:
    n_total: int = 80
    train_fraction: float = 0.80
    n_rep: int = 40
    seed_base: int = 20260527
    stationary_phi_true: float = 0.35
    stationary_alpha_true: float = 2.0
    nu_true: float = 4.0
    jitter: float = 1e-5
    spatial_scale: float = 1.0
    temporal_scale: float = 1.0

@dataclass
class FitConfig:
    hidden1: int = 32
    hidden2: int = 16
    auxiliary_hidden1: int = 16
    auxiliary_hidden2: int = 8
    iterations: int = 3000
    learning_rate: float = 1e-3
    posterior_draws: int = 5000
    surface_draws: int = 200
    grad_clip: float = 20.0
    kl_warmup_fraction: float = 0.25
    mean_pretrain_steps: int = 400
    mean_pretrain_lr: float = 0.01
    lr_decay_gamma: float = 0.2
    lr_decay_fraction1: float = 0.40
    lr_decay_fraction2: float = 0.70
    adaptive_fit_mode: str = "joint"
    svgp_inducing: int = 24
    svgp_learning_rate: float = 0.01
    adaptive_predictive_scale: float = 1.20
    early_stop_min_iter: int = 400
    early_stop_window: int = 150
    early_stop_check_every: int = 100
    early_stop_rel_slope: float = 1e-4
    early_stop_cv: float = 0.03
    print_every: int = 500
    dtype: str = "float64"

@dataclass
class ExperimentConfig:
    prior: PriorConfig = field(default_factory=PriorConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    fit: FitConfig = field(default_factory=FitConfig)

    def to_dict(self):
        return {
            "prior": asdict(self.prior),
            "simulation": asdict(self.simulation),
            "fit": asdict(self.fit),
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
