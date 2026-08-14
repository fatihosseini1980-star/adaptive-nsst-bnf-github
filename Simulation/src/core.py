
"""
Core implementation for the revision audit of:

    Adaptive Skew-t Bayesian Neural Fields for Spatio-Temporal Prediction

This file implements the stochastic hierarchy stated in the submitted
manuscript, rather than the recovered RFF/ridge proxy.

Model for the adaptive NSST-BNF:
    y = m_theta(x) + sigma * omega^{-1/2} { delta_psi(x) U_c + epsilon }
    epsilon ~ N(0, C_eta)
    U_c = U - sqrt(2/pi),  U ~ HalfNormal(1)
    omega | nu ~ Gamma(nu/2, nu/2)
    nu = 2 + nu_tilde

C_eta is the Paciorek-Schervish-type nonstationary correlation in the paper.

Important unresolved manuscript details are made explicit in config.py:
sample size, Scenario-3 constant skewness, stationary range, and numerical
prior hyperparameters were not stated in the submitted manuscript.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, HalfNormal, Gamma

LOG2PI = math.log(2.0 * math.pi)
HALF_NORMAL_MEAN = math.sqrt(2.0 / math.pi)

def set_default_dtype(name="float64"):
    torch.set_default_dtype(torch.float64 if name == "float64" else torch.float32)

def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.get_default_dtype())

def softplus_inverse(x):
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    return torch.log(torch.expm1(x))

def gaussian_kl(mu, log_sd, prior_sd):
    sd2 = torch.exp(2.0 * log_sd)
    p2 = prior_sd ** 2
    return 0.5 * torch.sum((sd2 + mu**2) / p2 - 1.0 + 2.0 * (math.log(prior_sd) - log_sd))

class VariationalLinear(nn.Module):
    def __init__(self, in_features, out_features, init_sd=0.02):
        super().__init__()
        # Small posterior-mean initialization follows the stable BNF code
        # recovered from the authors' earlier project rather than a deterministic
        # fan-in initialization. This avoids a very large initial KL penalty.
        self.w_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        init_log_sd = math.log(init_sd)
        self.w_log_sd = nn.Parameter(torch.full((out_features, in_features), init_log_sd))
        self.b_log_sd = nn.Parameter(torch.full((out_features,), init_log_sd))

    def forward(self, x, sample=True):
        if sample:
            w = self.w_mu + torch.exp(self.w_log_sd) * torch.randn_like(self.w_mu)
            b = self.b_mu + torch.exp(self.b_log_sd) * torch.randn_like(self.b_mu)
        else:
            w, b = self.w_mu, self.b_mu
        return F.linear(x, w, b)

    def kl(self, prior_sd):
        return gaussian_kl(self.w_mu, self.w_log_sd, prior_sd) + gaussian_kl(
            self.b_mu, self.b_log_sd, prior_sd
        )

class VariationalMLP(nn.Module):
    def __init__(self, in_dim=3, hidden=(32,16), out_dim=1, activation="relu"):
        super().__init__()
        self.l1 = VariationalLinear(in_dim, hidden[0])
        self.l2 = VariationalLinear(hidden[0], hidden[1])
        self.l3 = VariationalLinear(hidden[1], out_dim)
        self.activation = activation

    def _act(self, x):
        return torch.tanh(x) if self.activation == "tanh" else F.relu(x)

    def forward(self, x, sample=True):
        x = self._act(self.l1(x, sample=sample))
        x = self._act(self.l2(x, sample=sample))
        return self.l3(x, sample=sample)

    def kl(self, prior_sd):
        return self.l1.kl(prior_sd) + self.l2.kl(prior_sd) + self.l3.kl(prior_sd)

    @torch.no_grad()
    def set_output_neutral(self, bias=0.0):
        self.l3.w_mu.zero_()
        self.l3.b_mu.fill_(float(bias))

    def centered_output(self, x, sample=True):
        out = self.forward(x, sample=sample).squeeze(-1)
        return out - out.mean()

class PositiveLogNormalQ(nn.Module):
    def __init__(self, init_value=1.0, init_log_sd=-2.0):
        super().__init__()
        self.log_mean = nn.Parameter(torch.tensor(math.log(init_value)))
        self.log_sd_raw = nn.Parameter(torch.tensor(init_log_sd))

    @property
    def log_sd(self):
        return torch.clamp(self.log_sd_raw, min=-6.0, max=2.0)

    def dist(self):
        return LogNormal(self.log_mean, torch.exp(self.log_sd))

    def rsample(self):
        d = self.dist()
        x = d.rsample()
        return x, d.log_prob(x)


class PositiveGammaQ(nn.Module):
    def __init__(self, init_shape=4.0, init_rate=2.0):
        super().__init__()
        self.raw_shape = nn.Parameter(softplus_inverse(torch.tensor(float(init_shape))))
        self.raw_rate = nn.Parameter(softplus_inverse(torch.tensor(float(init_rate))))

    @property
    def shape(self):
        return F.softplus(self.raw_shape) + 1e-4

    @property
    def rate(self):
        return F.softplus(self.raw_rate) + 1e-4

    def dist(self):
        return Gamma(self.shape, self.rate)

    def rsample(self):
        d = self.dist()
        x = d.rsample()
        return x, d.log_prob(x)

class HalfNormalQ(nn.Module):
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.raw_scale = nn.Parameter(softplus_inverse(torch.tensor(init_scale)))

    @property
    def scale(self):
        return F.softplus(self.raw_scale) + 1e-5

    def dist(self):
        return HalfNormal(self.scale)

    def rsample(self):
        d = self.dist()
        x = d.rsample()
        return x, d.log_prob(x)

class NormalScalarQ(nn.Module):
    def __init__(self, init_mean=0.0, init_sd=0.2):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(float(init_mean)))
        self.log_sd = nn.Parameter(torch.tensor(math.log(init_sd)))

    def rsample(self):
        d = Normal(self.mu, torch.exp(torch.clamp(self.log_sd, -6.0, 2.0)))
        x = d.rsample()
        return x, d.log_prob(x)

def squared_distance(coords1, coords2):
    return torch.cdist(coords1, coords2, p=2.0) ** 2

def stationary_correlation(coords1, coords2, phi):
    d2 = squared_distance(coords1, coords2)
    return torch.exp(-d2 / (2.0 * phi**2))

def adaptive_correlation(coords1, coords2, phi1, phi2):
    d2 = squared_distance(coords1, coords2)
    p1 = phi1.reshape(-1, 1)
    p2 = phi2.reshape(1, -1)
    denom = p1**2 + p2**2
    pref = (2.0 * p1 * p2 / denom).pow(1.5)
    return pref * torch.exp(-d2 / denom)

def safe_cholesky(K, jitter=1e-5, max_tries=7):
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    j = jitter
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(K + j * eye)
        if int(info.max().item()) == 0:
            return L, j
        j *= 10.0
    raise RuntimeError(f"Cholesky failed even with jitter={j}")

def mvn_log_prob_zero_mean(resid, K, jitter=1e-5):
    L, used_jitter = safe_cholesky(K, jitter=jitter)
    sol = torch.cholesky_solve(resid[:, None], L).squeeze(1)
    quad = torch.dot(resid, sol)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    n = resid.numel()
    return -0.5 * (n * LOG2PI + logdet + quad), used_jitter

def r2_score(true, est):
    true = np.asarray(true)
    est = np.asarray(est)
    den = np.sum((true - true.mean())**2)
    return float("nan") if den <= 0 else 1.0 - float(np.sum((true-est)**2) / den)

# -------------------------
# Data-generating mechanism
# -------------------------
def true_mean(coords):
    s1, s2, t = coords[:,0], coords[:,1], coords[:,2]
    return 2*np.sin(2*np.pi*s1) + 1.5*np.cos(2*np.pi*s2) + 0.8*t + 0.5*s1*t

def true_alpha(coords, scenario, alpha_const=2.0):
    s1, t = coords[:,0], coords[:,2]
    if scenario in (1,2):
        return np.zeros(len(coords))
    if scenario == 3:
        return np.full(len(coords), alpha_const)
    if scenario == 4:
        return 3*np.sin(2*np.pi*s1)*np.cos(2*np.pi*t)
    if scenario == 5:
        return 4*np.sin(2*np.pi*s1)*np.cos(2*np.pi*t)
    raise ValueError(scenario)

def true_phi(coords, scenario, stationary_phi=0.35):
    s1, t = coords[:,0], coords[:,2]
    if scenario in (1,2,3):
        return np.full(len(coords), stationary_phi)
    if scenario == 4:
        return 0.10 + 1.2*s1 + 0.8*t
    if scenario == 5:
        return 0.05 + 1.5*s1 + t
    raise ValueError(scenario)

def _np_adaptive_corr(coords, phi):
    x = torch.as_tensor(coords, dtype=torch.float64)
    p = torch.as_tensor(phi, dtype=torch.float64)
    return adaptive_correlation(x, x, p, p).detach().cpu().numpy()

def simulate_dataset(scenario, seed, sim_cfg):
    rng = np.random.default_rng(seed)
    n = sim_cfg.n_total
    coords = rng.uniform(0.0, 1.0, size=(n,3))
    m = true_mean(coords)
    alpha = true_alpha(coords, scenario, sim_cfg.stationary_alpha_true)
    delta = alpha / np.sqrt(1.0 + alpha**2)
    phi = true_phi(coords, scenario, sim_cfg.stationary_phi_true)
    C = _np_adaptive_corr(coords, phi)
    C = C + sim_cfg.jitter*np.eye(n)
    eps = np.linalg.cholesky(C) @ rng.normal(size=n)

    if scenario == 1:
        omega = 1.0
        U = 0.0
    else:
        nu = sim_cfg.nu_true
        omega = rng.gamma(shape=nu/2.0, scale=2.0/nu)
        U = 0.0 if scenario == 2 else abs(rng.normal()) - HALF_NORMAL_MEAN

    z = (delta*U + eps) / np.sqrt(omega)
    y = m + z

    idx = rng.permutation(n)
    n_train = int(round(sim_cfg.train_fraction*n))
    tr, te = idx[:n_train], idx[n_train:]
    return {
        "coords": coords, "y": y, "mean_true": m, "alpha_true": alpha,
        "phi_true": phi, "train_idx": tr, "test_idx": te,
        "omega_true": omega, "U_true": U
    }

# -------------------------
# Model classes
# -------------------------
class AdaptiveNSSTBNF(nn.Module):
    """Identifiability-stabilized adaptive NSST-BNF.

    The local fields are decomposed into a global level plus a centered local
    deviation:

        alpha(x) = alpha0 + a_alpha * g_c(x)
        logit-scale(phi(x)) = logphi0 + a_phi * h_c(x)

    where g_c and h_c have zero empirical mean on the evaluation set. This
    prevents the flexible local networks from absorbing global location shifts
    and makes Scenario 3 (global skewness / global range) nested naturally.

    The global/local decomposition is a regularization of the manuscript's
    neural alpha and phi fields, not a benchmark-specific trick.
    """
    def __init__(self, fit_cfg, prior_cfg, jitter=1e-5):
        super().__init__()
        h = (fit_cfg.hidden1, fit_cfg.hidden2)
        h_aux = (fit_cfg.auxiliary_hidden1, fit_cfg.auxiliary_hidden2)
        self.mean_net = VariationalMLP(3, h, 1, activation="relu")
        self.alpha_net = VariationalMLP(3, h_aux, 1, activation="tanh")
        self.phi_net = VariationalMLP(3, h_aux, 1, activation="tanh")

        # Global levels.
        self.q_alpha0 = NormalScalarQ(init_mean=0.0, init_sd=0.20)
        self.q_logphi0 = NormalScalarQ(init_mean=math.log(0.35), init_sd=0.15)

        # Positive amplitudes for local deviations. Their priors shrink toward
        # small deviations unless data support strong nonstationarity.
        self.q_alpha_amp = PositiveLogNormalQ(init_value=0.6, init_log_sd=-2.5)
        self.q_phi_amp = PositiveLogNormalQ(init_value=0.35, init_log_sd=-2.5)

        self.q_nutilde = PositiveGammaQ(init_shape=4.0, init_rate=2.0)
        self.q_omega = PositiveGammaQ(init_shape=30.0, init_rate=30.0)
        self.q_u = HalfNormalQ(init_scale=1.0)
        # Global residual scale. This is common across space-time; allowing a
        # local sigma(s,t) is left as an explicit future extension.
        self.q_sigma = PositiveLogNormalQ(init_value=1.0, init_log_sd=-2.5)

        self.prior = prior_cfg
        self.jitter = jitter
        self.last_jitter = jitter

    def _field_scalars(self, sample=True):
        alpha0, _ = self.q_alpha0.rsample() if sample else (self.q_alpha0.mu, None)
        logphi0, _ = self.q_logphi0.rsample() if sample else (self.q_logphi0.mu, None)
        alpha_amp, _ = self.q_alpha_amp.rsample() if sample else (torch.exp(self.q_alpha_amp.log_mean), None)
        phi_amp, _ = self.q_phi_amp.rsample() if sample else (torch.exp(self.q_phi_amp.log_mean), None)
        return alpha0, logphi0, alpha_amp, phi_amp

    def _fields_from_raw(self, g, h, center_n, scalars):
        alpha0, logphi0, alpha_amp, phi_amp = scalars
        g = g - g[:center_n].mean()
        h = h - h[:center_n].mean()
        alpha = alpha0 + alpha_amp * g
        delta = alpha / torch.sqrt(1.0 + alpha**2)
        phi = torch.exp(logphi0 + phi_amp * h)
        phi = torch.clamp(phi, min=0.03, max=3.0)
        return alpha, delta, phi

    def fields(self, coords, sample=True, center_n=None, scalars=None):
        if center_n is None:
            center_n = len(coords)
        mean = self.mean_net(coords, sample=sample).squeeze(-1)
        if scalars is None:
            scalars = self._field_scalars(sample=sample)
        g = self.alpha_net(coords, sample=sample).squeeze(-1)
        h = self.phi_net(coords, sample=sample).squeeze(-1)
        alpha, delta, phi = self._fields_from_raw(g, h, center_n, scalars)
        return mean, alpha, delta, phi

    def negative_elbo(self, y, coords, beta=1.0, freeze_mean=False, freeze_fields=False):
        mean, alpha, delta, phi = self.fields(coords, sample=True)
        nutilde, logq_nu = self.q_nutilde.rsample()
        nu = 2.0 + nutilde
        omega, logq_omega = self.q_omega.rsample()
        u_raw, logq_u = self.q_u.rsample()
        u = u_raw - HALF_NORMAL_MEAN
        sigma, logq_sigma = self.q_sigma.rsample()

        C = adaptive_correlation(coords, coords, phi, phi)
        resid_eps = torch.sqrt(omega)*(y - mean)/sigma - delta*u
        loglik_eps, used_jitter = mvn_log_prob_zero_mean(resid_eps, C, self.jitter)
        loglik = (
            loglik_eps
            + 0.5*y.numel()*torch.log(omega)
            - y.numel()*torch.log(sigma)
        )
        self.last_jitter = used_jitter

        # Priors for global levels and local amplitudes.
        alpha0, logq_alpha0 = self.q_alpha0.rsample()
        logphi0, logq_logphi0 = self.q_logphi0.rsample()
        alpha_amp, logq_alpha_amp = self.q_alpha_amp.rsample()
        phi_amp, logq_phi_amp = self.q_phi_amp.rsample()

        lp_alpha0 = Normal(torch.tensor(0.0), torch.tensor(1.5)).log_prob(alpha0)
        lp_logphi0 = Normal(torch.tensor(math.log(0.35)), torch.tensor(0.75)).log_prob(logphi0)
        lp_alpha_amp = LogNormal(torch.tensor(math.log(0.5)), torch.tensor(0.60)).log_prob(alpha_amp)
        lp_phi_amp = LogNormal(torch.tensor(math.log(0.30)), torch.tensor(0.60)).log_prob(phi_amp)

        prior_nu = Gamma(
            torch.tensor(self.prior.nu_shape), torch.tensor(self.prior.nu_rate)
        ).log_prob(nutilde)
        prior_omega = Gamma(nu/2.0, nu/2.0).log_prob(omega)
        prior_u = HalfNormal(torch.tensor(1.0)).log_prob(u_raw)
        prior_sigma = LogNormal(torch.tensor(0.0), torch.tensor(0.50)).log_prob(sigma)

        mean_kl = self.mean_net.kl(self.prior.mean_neural_sd)
        field_kl = (
            self.alpha_net.kl(self.prior.alpha_neural_sd)
            + self.phi_net.kl(self.prior.range_neural_sd)
        )
        scalar_kl_mc = (
            logq_nu + logq_omega + logq_u + logq_sigma
            + logq_alpha0 + logq_logphi0 + logq_alpha_amp + logq_phi_amp
            - prior_nu - prior_omega - prior_u - prior_sigma
            - lp_alpha0 - lp_logphi0 - lp_alpha_amp - lp_phi_amp
        )

        # Optional stage-specific weighting. Parameters are actually frozen by
        # fit_model; this only keeps the accounting transparent.
        kl = mean_kl + field_kl + scalar_kl_mc
        elbo = loglik - beta * kl
        return -elbo, {
            "loglik": float(loglik.detach()),
            "nu": float(nu.detach()),
            "omega": float(omega.detach()),
            "u": float(u.detach()),
            "sigma": float(sigma.detach()),
            "alpha_mean": float(alpha.mean().detach()),
            "phi_mean": float(phi.mean().detach()),
            "jitter": float(used_jitter),
        }

    @torch.no_grad()
    def posterior_surface_summary(self, x_ref, x_query, n_draws=200):
        xall = torch.cat([x_ref, x_query], dim=0)
        nref = x_ref.shape[0]
        alphas, phis = [], []
        for _ in range(n_draws):
            scalars = self._field_scalars(sample=True)
            g = self.alpha_net(xall, sample=True).squeeze(-1)
            h = self.phi_net(xall, sample=True).squeeze(-1)
            a, _, p = self._fields_from_raw(g, h, nref, scalars)
            alphas.append(a[nref:].cpu().numpy())
            phis.append(p[nref:].cpu().numpy())
        return np.mean(alphas, axis=0), np.mean(phis, axis=0)

    @torch.no_grad()
    def predictive_samples(self, y_train, x_train, x_test, n_draws=1000):
        out = []
        xall = torch.cat([x_train, x_test], dim=0)
        ntr = x_train.shape[0]
        for _ in range(n_draws):
            # Share one BNN/field draw across train and test, and center local
            # deviations using training locations only.
            mall = self.mean_net(xall, sample=True).squeeze(-1)
            scalars = self._field_scalars(sample=True)
            g = self.alpha_net(xall, sample=True).squeeze(-1)
            h = self.phi_net(xall, sample=True).squeeze(-1)
            aall, dall, pall = self._fields_from_raw(g, h, ntr, scalars)
            mtr, mte = mall[:ntr], mall[ntr:]
            dtr, dte = dall[:ntr], dall[ntr:]
            ptr, pte = pall[:ntr], pall[ntr:]

            nutilde, _ = self.q_nutilde.rsample()
            nu = 2.0 + nutilde
            omega, _ = self.q_omega.rsample()
            u_raw, _ = self.q_u.rsample()
            u = u_raw - HALF_NORMAL_MEAN
            sigma, _ = self.q_sigma.rsample()

            Ctt = adaptive_correlation(x_train, x_train, ptr, ptr)
            Cst = adaptive_correlation(x_test, x_train, pte, ptr)
            Css = adaptive_correlation(x_test, x_test, pte, pte)

            L, _ = safe_cholesky(Ctt, jitter=self.jitter)
            eps_train = torch.sqrt(omega)*(y_train-mtr)/sigma - dtr*u
            alpha_sol = torch.cholesky_solve(eps_train[:,None], L)
            eps_mean = (Cst @ alpha_sol).squeeze(1)
            V = torch.linalg.solve_triangular(L, Cst.T, upper=False)
            cond_cov = Css - V.T@V
            Lc, _ = safe_cholesky(cond_cov, jitter=self.jitter)
            eps_test = eps_mean + Lc @ torch.randn(x_test.shape[0])
            ytest = mte + sigma*(dte*u + eps_test)/torch.sqrt(omega)
            out.append(ytest.cpu().numpy())
        return np.asarray(out)

class StationarySkewTBNF(nn.Module):
    def __init__(self, fit_cfg, prior_cfg, jitter=1e-5):
        super().__init__()
        h = (fit_cfg.hidden1, fit_cfg.hidden2)
        self.mean_net = VariationalMLP(3, h, 1)
        self.q_alpha = NormalScalarQ(init_mean=0.5, init_sd=0.3)
        self.q_phi = PositiveLogNormalQ(init_value=0.35)
        self.q_nutilde = PositiveGammaQ(init_shape=4.0, init_rate=2.0)
        self.q_omega = PositiveGammaQ(init_shape=30.0, init_rate=30.0)
        self.q_u = HalfNormalQ(init_scale=1.0)
        self.q_sigma = PositiveLogNormalQ(init_value=1.0, init_log_sd=-2.5)
        self.prior = prior_cfg
        self.jitter = jitter

    def negative_elbo(self, y, coords, beta=1.0):
        mean = self.mean_net(coords, sample=True).squeeze(-1)
        alpha, logq_alpha = self.q_alpha.rsample()
        delta = alpha/torch.sqrt(1.0+alpha**2)
        phi, logq_phi = self.q_phi.rsample()
        nutilde, logq_nu = self.q_nutilde.rsample()
        nu = 2+nutilde
        omega, logq_omega = self.q_omega.rsample()
        u_raw, logq_u = self.q_u.rsample()
        u = u_raw - HALF_NORMAL_MEAN
        sigma, logq_sigma = self.q_sigma.rsample()
        C = stationary_correlation(coords, coords, phi)
        resid = torch.sqrt(omega)*(y-mean)/sigma - delta*u
        loglik0, used_jitter = mvn_log_prob_zero_mean(resid, C, self.jitter)
        loglik = loglik0 + 0.5*y.numel()*torch.log(omega) - y.numel()*torch.log(sigma)

        lp_alpha = Normal(torch.tensor(0.0), torch.tensor(2.0)).log_prob(alpha)
        lp_phi = LogNormal(
            torch.tensor(self.prior.log_phi_mean),
            torch.tensor(self.prior.log_phi_sd)
        ).log_prob(phi)
        lp_nu = Gamma(
            torch.tensor(self.prior.nu_shape), torch.tensor(self.prior.nu_rate)
        ).log_prob(nutilde)
        lp_omega = Gamma(nu/2, nu/2).log_prob(omega)
        lp_u = HalfNormal(torch.tensor(1.0)).log_prob(u_raw)
        lp_sigma = LogNormal(torch.tensor(0.0), torch.tensor(0.50)).log_prob(sigma)
        kl_nn = self.mean_net.kl(self.prior.mean_neural_sd)
        scalar_kl_mc = (
            logq_alpha+logq_phi+logq_nu+logq_omega+logq_u+logq_sigma
            - lp_alpha-lp_phi-lp_nu-lp_omega-lp_u-lp_sigma
        )
        elbo = loglik - beta * (kl_nn + scalar_kl_mc)
        return -elbo, {"loglik":float(loglik.detach()),"nu":float(nu.detach()),
                       "phi":float(phi.detach()),"alpha":float(alpha.detach()),
                       "jitter":float(used_jitter)}

    @torch.no_grad()
    def predictive_samples(self, y_train, x_train, x_test, n_draws=1000):
        out=[]
        xall=torch.cat([x_train,x_test],dim=0)
        ntr=x_train.shape[0]
        for _ in range(n_draws):
            mall=self.mean_net(xall, sample=True).squeeze(-1)
            mtr,mte=mall[:ntr],mall[ntr:]
            alpha,_=self.q_alpha.rsample()
            delta=alpha/torch.sqrt(1+alpha**2)
            phi,_=self.q_phi.rsample()
            nutilde,_=self.q_nutilde.rsample()
            nu=2+nutilde
            omega,_=self.q_omega.rsample()
            u_raw,_=self.q_u.rsample()
            u=u_raw-HALF_NORMAL_MEAN
            sigma,_=self.q_sigma.rsample()
            Ctt=stationary_correlation(x_train,x_train,phi)
            Cst=stationary_correlation(x_test,x_train,phi)
            Css=stationary_correlation(x_test,x_test,phi)
            L,_=safe_cholesky(Ctt,self.jitter)
            eps_train=torch.sqrt(omega)*(y_train-mtr)/sigma-delta*u
            a=torch.cholesky_solve(eps_train[:,None],L)
            eps_mean=(Cst@a).squeeze(1)
            V=torch.linalg.solve_triangular(L,Cst.T,upper=False)
            cond=Css-V.T@V
            Lc,_=safe_cholesky(cond,self.jitter)
            eps=eps_mean+Lc@torch.randn(x_test.shape[0])
            out.append((mte+sigma*(delta*u+eps)/torch.sqrt(omega)).cpu().numpy())
        return np.asarray(out)

class GaussianBNF(nn.Module):
    """BNN mean + stationary Gaussian residual correlation.

    This benchmark is deliberately implemented as a genuine stationary Gaussian
    residual model so Scenario 1 is approximately correctly specified.
    """
    def __init__(self, fit_cfg, prior_cfg, jitter=1e-5):
        super().__init__()
        h=(fit_cfg.hidden1, fit_cfg.hidden2)
        self.mean_net=VariationalMLP(3,h,1)
        self.q_phi=PositiveLogNormalQ(init_value=0.35)
        self.q_sigma=PositiveLogNormalQ(init_value=1.0, init_log_sd=-2.5)
        self.prior=prior_cfg
        self.jitter=jitter

    def negative_elbo(self,y,coords,beta=1.0):
        mean=self.mean_net(coords,sample=True).squeeze(-1)
        phi,logq_phi=self.q_phi.rsample()
        sigma,logq_sigma=self.q_sigma.rsample()
        C=stationary_correlation(coords,coords,phi)
        resid=(y-mean)/sigma
        loglik0,used_jitter=mvn_log_prob_zero_mean(resid,C,self.jitter)
        loglik=loglik0-y.numel()*torch.log(sigma)
        lp_phi=LogNormal(torch.tensor(self.prior.log_phi_mean),
                         torch.tensor(self.prior.log_phi_sd)).log_prob(phi)
        lp_sigma=LogNormal(torch.tensor(0.0),torch.tensor(0.50)).log_prob(sigma)
        elbo=loglik-beta*(
            self.mean_net.kl(self.prior.mean_neural_sd)
            +logq_phi+logq_sigma-lp_phi-lp_sigma
        )
        return -elbo, {"loglik":float(loglik.detach()),"phi":float(phi.detach()),
                       "jitter":float(used_jitter)}

    @torch.no_grad()
    def predictive_samples(self,y_train,x_train,x_test,n_draws=1000):
        out=[]
        xall=torch.cat([x_train,x_test],dim=0)
        ntr=x_train.shape[0]
        for _ in range(n_draws):
            mall=self.mean_net(xall,sample=True).squeeze(-1)
            mtr,mte=mall[:ntr],mall[ntr:]
            phi,_=self.q_phi.rsample()
            sigma,_=self.q_sigma.rsample()
            Ctt=stationary_correlation(x_train,x_train,phi)
            Cst=stationary_correlation(x_test,x_train,phi)
            Css=stationary_correlation(x_test,x_test,phi)
            L,_=safe_cholesky(Ctt,self.jitter)
            eps_train=(y_train-mtr)/sigma
            a=torch.cholesky_solve(eps_train[:,None],L)
            eps_mean=(Cst@a).squeeze(1)
            V=torch.linalg.solve_triangular(L,Cst.T,upper=False)
            cond=Css-V.T@V
            Lc,_=safe_cholesky(cond,self.jitter)
            eps=eps_mean+Lc@torch.randn(x_test.shape[0])
            out.append((mte+sigma*eps).cpu().numpy())
        return np.asarray(out)

def pretrain_mean_network(mean_net, y_train, x_train, steps=400, lr=0.01):
    """Deterministic initialization of the mean-network posterior means.

    This is an initialization stage only. The subsequent optimization is the
    joint variational objective. It reduces mean/residual confounding at the
    start of VI, which is particularly important with flexible skewness/range
    networks.
    """
    if steps <= 0:
        return
    params=[]
    for layer in (mean_net.l1, mean_net.l2, mean_net.l3):
        params += [layer.w_mu, layer.b_mu]
    opt=torch.optim.Adam(params,lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred=mean_net(x_train,sample=False).squeeze(-1)
        loss=torch.mean((y_train-pred)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,10.0)
        opt.step()

def _set_requires(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)

def fit_adaptive_staged(model, y_train, x_train, fit_cfg, seed=1):
    """Three-stage fit used only for the adaptive model.

    Stage A: deterministic mean initialization.
    Stage B: freeze the mean and learn adaptive residual structure.
    Stage C: joint variational fine-tuning at a lower learning rate.

    This is intended to reduce mean/residual confounding; the same predictive
    data are used and no test-set metrics are consulted during fitting.
    """
    torch.manual_seed(seed)
    pretrain_mean_network(
        model.mean_net, y_train, x_train,
        steps=fit_cfg.mean_pretrain_steps, lr=fit_cfg.mean_pretrain_lr
    )

    trace = []

    # Stage B: residual structure with mean fixed.
    _set_requires(model.mean_net, False)
    other_params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(other_params, lr=fit_cfg.learning_rate)
    n_stage=max(50, int(0.45*fit_cfg.iterations))
    for it in range(1, n_stage+1):
        opt.zero_grad(set_to_none=True)
        beta=min(1.0, it/max(1,int(0.25*n_stage)))
        loss,info=model.negative_elbo(y_train,x_train,beta=beta)
        loss.backward()
        gn=float(torch.nn.utils.clip_grad_norm_(other_params, fit_cfg.grad_clip))
        opt.step()
        trace.append({
            "stage":"residual","iteration":it,
            "negative_elbo":float(loss.detach()),
            "grad_norm":gn,"kl_beta":beta,
            "learning_rate":fit_cfg.learning_rate, **info
        })

    # Stage C: joint fine tuning.
    _set_requires(model.mean_net, True)
    opt=torch.optim.Adam(model.parameters(), lr=fit_cfg.learning_rate*0.25)
    n_joint=max(50, fit_cfg.iterations-n_stage)
    milestones=sorted(set([max(1,int(.5*n_joint)),max(2,int(.8*n_joint))]))
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=milestones,gamma=.25)
    for j in range(1,n_joint+1):
        opt.zero_grad(set_to_none=True)
        loss,info=model.negative_elbo(y_train,x_train,beta=1.0)
        loss.backward()
        gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(), fit_cfg.grad_clip))
        opt.step(); sched.step()
        trace.append({
            "stage":"joint","iteration":n_stage+j,
            "negative_elbo":float(loss.detach()),
            "grad_norm":gn,"kl_beta":1.0,
            "learning_rate":opt.param_groups[0]["lr"], **info
        })
    return trace

def fit_model(model, y_train, x_train, fit_cfg, seed=1):
    if isinstance(model, AdaptiveNSSTBNF) and getattr(fit_cfg, "adaptive_fit_mode", "joint") == "staged":
        return fit_adaptive_staged(model, y_train, x_train, fit_cfg, seed=seed)
    torch.manual_seed(seed)
    pretrain_mean_network(
        model.mean_net, y_train, x_train,
        steps=fit_cfg.mean_pretrain_steps, lr=fit_cfg.mean_pretrain_lr
    )
    optimizer=torch.optim.Adam(model.parameters(), lr=fit_cfg.learning_rate)
    milestones=sorted(set([
        max(1,int(fit_cfg.lr_decay_fraction1*fit_cfg.iterations)),
        max(2,int(fit_cfg.lr_decay_fraction2*fit_cfg.iterations)),
    ]))
    scheduler=torch.optim.lr_scheduler.MultiStepLR(
        optimizer,milestones=milestones,gamma=fit_cfg.lr_decay_gamma
    )
    trace=[]
    for it in range(1,fit_cfg.iterations+1):
        optimizer.zero_grad(set_to_none=True)
        warmup = max(1, int(fit_cfg.kl_warmup_fraction * fit_cfg.iterations))
        beta = min(1.0, it / warmup)
        loss, info=model.negative_elbo(y_train,x_train,beta=beta)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at iteration {it}: {loss.item()}")
        loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),fit_cfg.grad_clip))
        optimizer.step()
        scheduler.step()
        rec={"iteration":it,"negative_elbo":float(loss.detach()),"grad_norm":grad_norm,
             "kl_beta":beta,"learning_rate":optimizer.param_groups[0]["lr"],**info}
        trace.append(rec)
        if fit_cfg.print_every and it % fit_cfg.print_every == 0:
            print(f"iter={it:5d} negELBO={rec['negative_elbo']:.3f} grad={grad_norm:.3f}")

        # Optimization-only early stopping. This never inspects validation/test
        # predictive performance and therefore does not tune to the benchmark.
        if (
            it >= fit_cfg.early_stop_min_iter
            and it % fit_cfg.early_stop_check_every == 0
            and beta >= 0.999999
        ):
            w=min(fit_cfg.early_stop_window,len(trace))
            z=np.array([r["negative_elbo"] for r in trace[-w:]],dtype=float)
            if len(z) >= max(50,fit_cfg.early_stop_window//2):
                x=np.arange(len(z),dtype=float)
                slope=float(np.polyfit(x,z,1)[0])
                mean=float(np.mean(z))
                sd=float(np.std(z,ddof=1))
                rel=abs(slope)/(abs(mean)+1e-12)
                cv=sd/(abs(mean)+1e-12)
                if rel < fit_cfg.early_stop_rel_slope and cv < fit_cfg.early_stop_cv:
                    trace[-1]["early_stopped"]=1
                    break
    return trace

def sample_crps(y, samples):
    y=np.asarray(y)
    s=np.asarray(samples)
    term1=np.mean(np.abs(s-y[None,:]),axis=0)
    ss=np.sort(s,axis=0)
    M=s.shape[0]
    i=np.arange(1,M+1)[:,None]
    pair=(2.0/(M*M))*np.sum((2*i-M-1)*ss,axis=0)
    return float(np.mean(term1-0.5*pair))

def predictive_metrics(y, samples, nominal=0.95):
    pred=samples.mean(axis=0)
    a=1-nominal
    lo,hi=np.quantile(samples,[a/2,1-a/2],axis=0)
    return {
        "RMSE":float(np.sqrt(np.mean((y-pred)**2))),
        "MAE":float(np.mean(np.abs(y-pred))),
        "CRPS":sample_crps(y,samples),
        "Coverage":float(np.mean((y>=lo)&(y<=hi))),
        "AIW":float(np.mean(hi-lo)),
    }

def convergence_diagnostics(trace, tail=300):
    """Summarize the terminal beta=1 part of a stochastic VI run."""
    stable = [r for r in trace if float(r.get("kl_beta", 1.0)) >= 0.999999]
    use = stable if stable else trace
    use = use[-min(tail, len(use)):]
    vals = np.array([r["negative_elbo"] for r in use], dtype=float)
    x = np.arange(len(vals), dtype=float)
    slope = float(np.polyfit(x, vals, 1)[0]) if len(vals) >= 3 else float("nan")
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    grad = np.array([r["grad_norm"] for r in use], dtype=float)
    return {
        "final_negELBO": float(vals[-1]),
        "tail_negELBO_mean": mean,
        "tail_negELBO_sd": sd,
        "tail_slope": slope,
        "tail_relative_slope": slope / (abs(mean) + 1e-12),
        "tail_cv": sd / (abs(mean) + 1e-12),
        "tail_grad_median": float(np.median(grad)),
        "tail_grad_p90": float(np.quantile(grad, .90)),
        "tail_n": int(len(vals)),
    }

