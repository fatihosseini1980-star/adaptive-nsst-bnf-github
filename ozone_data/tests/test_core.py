import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from config import ExperimentConfig
from core import (
    set_default_dtype, adaptive_correlation, strict_cholesky,
    select_calibration_multiplier, rescale_predictive_dispersion,
    GaussianBNF, AdaptiveNSSTBNF,
)

set_default_dtype("float64")


def test_adaptive_correlation_symmetry_and_diagonal():
    x = torch.tensor([[0.,0.,0.],[.2,.5,.1],[.9,.3,.7]])
    p = torch.tensor([.2,.4,.8])
    C = adaptive_correlation(x,x,p,p)
    assert torch.allclose(C, C.T, atol=1e-12)
    assert torch.allclose(torch.diag(C), torch.ones(3), atol=1e-12)
    strict_cholesky(C, 1e-10, 1e-4, 10.)


def test_calibration_can_shrink():
    rng = np.random.default_rng(1)
    y = np.zeros(200)
    draws = rng.normal(0, 3.0, size=(500, 200))
    m, _ = select_calibration_multiplier(y, draws, nominal=.95, lo=.3, hi=2.0, step=.05)
    assert m < 1.0


def test_gaussian_prediction_uses_joint_mean_draw_shape():
    cfg = ExperimentConfig()
    m = GaussianBNF(5, cfg.fit, cfg.prior, cfg.numerical)
    assert m.mean_net.l1.w_mu.shape[1] == 5


def test_adaptive_field_train_reference_invariance():
    cfg = ExperimentConfig()
    torch.manual_seed(7)
    m = AdaptiveNSSTBNF(5,3,cfg.fit,cfg.prior,cfg.numerical)
    ref = torch.rand(8,3)
    q1 = torch.rand(3,3)
    q2 = torch.cat([q1, torch.rand(4,3)], 0)
    # deterministic posterior means: first three query field values must not
    # change when unrelated query points are appended.
    def evalq(q):
        allx = torch.cat([ref,q],0)
        g = m.alpha_net(allx, sample=False).squeeze(-1)
        h = m.phi_net(allx, sample=False).squeeze(-1)
        scalars = (m.q_alpha0.mu, m.q_logphi0.mu,
                   torch.exp(m.q_alpha_amp.log_mean), torch.exp(m.q_phi_amp.log_mean))
        a,_,p = m._fields_from_raw(g,h,len(ref),scalars)
        return a[len(ref):], p[len(ref):]
    a1,p1 = evalq(q1); a2,p2 = evalq(q2)
    assert torch.allclose(a1, a2[:3], atol=1e-12)
    assert torch.allclose(p1, p2[:3], atol=1e-12)
