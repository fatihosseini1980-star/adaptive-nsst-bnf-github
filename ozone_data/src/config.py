from dataclasses import dataclass, field, asdict
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
class NumericalConfig:
    # Numerical stabilization only. The routine fails rather than silently
    # escalating beyond max_jitter, so jitter cannot become an implicit nugget.
    initial_jitter: float = 1e-8
    max_jitter: float = 1e-4
    jitter_multiplier: float = 10.0
    dtype: str = "float64"


@dataclass
class FitConfig:
    hidden1: int = 32
    hidden2: int = 16
    auxiliary_hidden1: int = 16
    auxiliary_hidden2: int = 8
    iterations_gaussian: int = 900
    iterations_adaptive: int = 900
    learning_rate: float = 8e-4
    posterior_draws: int = 500
    surface_draws: int = 100
    grad_clip: float = 20.0
    kl_warmup_fraction: float = 0.20
    mean_pretrain_steps: int = 500
    mean_pretrain_lr: float = 8e-3
    adaptive_stage_fraction: float = 0.47
    adaptive_joint_lr_factor: float = 0.25
    restart_seeds: tuple = (111, 222, 333)
    objective_tail: int = 60


@dataclass
class OzoneConfig:
    panel_size: int = 24
    n_test_stations: int = 6
    n_calibration_stations: int = 4
    n_days: int = 16
    nominal_coverage: float = 0.95
    calibration_min: float = 0.50
    calibration_max: float = 3.00
    calibration_step: float = 0.05
    # QC rules are response-independent except for obvious structural anomalies:
    # stations with >20% exact zeros or <50% observed days are excluded.
    zero_fraction_max: float = 0.20
    observed_fraction_min: float = 0.50


@dataclass
class ExperimentConfig:
    prior: PriorConfig = field(default_factory=PriorConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    ozone: OzoneConfig = field(default_factory=OzoneConfig)

    def to_dict(self):
        return {
            "prior": asdict(self.prior),
            "numerical": asdict(self.numerical),
            "fit": asdict(self.fit),
            "ozone": asdict(self.ozone),
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
