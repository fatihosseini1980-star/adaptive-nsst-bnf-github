import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ExperimentConfig
from core import set_default_dtype, simulate_dataset, to_tensor, AdaptiveNSSTBNF, GaussianBNF


def test_all_scenarios_generate_expected_shapes():
    cfg = ExperimentConfig()
    cfg.simulation.n_total = 30
    for s in range(1, 6):
        d = simulate_dataset(s, 12345 + s, cfg.simulation)
        assert d["coords"].shape == (30, 3)
        assert d["y"].shape == (30,)
        assert len(d["train_idx"]) + len(d["test_idx"]) == 30


def test_adaptive_field_centering_uses_reference_prefix():
    set_default_dtype("float64")
    cfg = ExperimentConfig()
    model = AdaptiveNSSTBNF(cfg.fit, cfg.prior, cfg.simulation.jitter)
    x = torch.tensor(np.linspace(0, 1, 24).reshape(8, 3), dtype=torch.float64)
    with torch.no_grad():
        scalars = model._field_scalars(sample=False)
        g = model.alpha_net(x, sample=False).squeeze(-1)
        h = model.phi_net(x, sample=False).squeeze(-1)
        a, _, p = model._fields_from_raw(g, h, 5, scalars)
        # Reconstruct the centered local components; their reference-prefix mean is zero.
        alpha0, logphi0, alpha_amp, phi_amp = scalars
        if float(alpha_amp) > 0:
            local_a = (a - alpha0) / alpha_amp
            assert abs(float(local_a[:5].mean())) < 1e-10
        local_h = (torch.log(p) - logphi0) / phi_amp
        assert abs(float(local_h[:5].mean())) < 1e-8


def test_gaussian_predictive_shape():
    set_default_dtype("float64")
    cfg = ExperimentConfig()
    cfg.simulation.n_total = 24
    d = simulate_dataset(1, 999, cfg.simulation)
    tr, te = d["train_idx"], d["test_idx"]
    model = GaussianBNF(cfg.fit, cfg.prior, cfg.simulation.jitter)
    out = model.predictive_samples(
        to_tensor(d["y"][tr]), to_tensor(d["coords"][tr]), to_tensor(d["coords"][te]), n_draws=3
    )
    assert out.shape == (3, len(te))
    assert np.isfinite(out).all()


def test_frozen_predictive_scale_is_1p20():
    cfg = ExperimentConfig()
    assert abs(cfg.fit.adaptive_predictive_scale - 1.20) < 1e-12
