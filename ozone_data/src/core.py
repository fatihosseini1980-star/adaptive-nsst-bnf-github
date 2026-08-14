"""Core implementation of the two models released with this repository.

The code intentionally contains only:
  1. Gaussian BNF reference model
  2. Adaptive NSST-BNF proposed model

Important numerical rules
-------------------------
* Jitter is numerical stabilization, not an observation-noise/nugget model.
* Cholesky jitter is capped. If the cap is exceeded, fitting stops explicitly.
* Within one posterior predictive draw, the same sampled BNN weights are used
  for both training and prediction locations.
* Adaptive-field centering is defined by the training locations only. Test
  locations never enter the centering reference.
"""
import math
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, HalfNormal, Gamma

LOG2PI = math.log(2.0 * math.pi)
HALF_NORMAL_MEAN = math.sqrt(2.0 / math.pi)


def set_default_dtype(name="float64"):
    torch.set_default_dtype(torch.float64 if name == "float64" else torch.float32)


def set_all_seeds(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


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
    def __init__(self, in_dim, hidden=(32, 16), out_dim=1, activation="relu"):
        super().__init__()
        self.l1 = VariationalLinear(in_dim, hidden[0])
        self.l2 = VariationalLinear(hidden[0], hidden[1])
        self.l3 = VariationalLinear(hidden[1], out_dim)
        self.activation = activation

    def _act(self, x):
        return torch.tanh(x) if self.activation == "tanh" else F.relu(x)

    def forward(self, x, sample=True):
        z = self._act(self.l1(x, sample=sample))
        z = self._act(self.l2(z, sample=sample))
        return self.l3(z, sample=sample)

    def kl(self, prior_sd):
        return self.l1.kl(prior_sd) + self.l2.kl(prior_sd) + self.l3.kl(prior_sd)

    @torch.no_grad()
    def set_output_neutral(self, bias=0.0):
        self.l3.w_mu.zero_()
        self.l3.b_mu.fill_(float(bias))


class PositiveLogNormalQ(nn.Module):
    def __init__(self, init_value=1.0, init_log_sd=-2.5):
        super().__init__()
        self.log_mean = nn.Parameter(torch.tensor(math.log(float(init_value))))
        self.log_sd_raw = nn.Parameter(torch.tensor(float(init_log_sd)))

    @property
    def log_sd(self):
        return torch.clamp(self.log_sd_raw, min=-6.0, max=2.0)

    def rsample(self):
        d = LogNormal(self.log_mean, torch.exp(self.log_sd))
        x = d.rsample()
        return x, d.log_prob(x)

    def median(self):
        return torch.exp(self.log_mean)


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

    def rsample(self):
        d = Gamma(self.shape, self.rate)
        x = d.rsample()
        return x, d.log_prob(x)


class HalfNormalQ(nn.Module):
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.raw_scale = nn.Parameter(softplus_inverse(torch.tensor(float(init_scale))))

    @property
    def scale(self):
        return F.softplus(self.raw_scale) + 1e-5

    def rsample(self):
        d = HalfNormal(self.scale)
        x = d.rsample()
        return x, d.log_prob(x)


class NormalScalarQ(nn.Module):
    def __init__(self, init_mean=0.0, init_sd=0.2):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(float(init_mean)))
        self.log_sd = nn.Parameter(torch.tensor(math.log(float(init_sd))))

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
    pref = (2.0 * p1 * p2 / denom).pow(1.5)  # d/2 with d=3
    return pref * torch.exp(-d2 / denom)


class NumericalStabilityError(RuntimeError):
    pass


def strict_cholesky(K, initial_jitter=1e-8, max_jitter=1e-4, multiplier=10.0):
    """Cholesky with bounded numerical stabilization.

    The matrix is symmetrized first. Jitter may increase only up to
    ``max_jitter``. Exceeding that cap raises an error rather than silently
    changing the stochastic model into one with a substantial nugget.
    """
    K = 0.5 * (K + K.T)
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    j = float(initial_jitter)
    while j <= float(max_jitter) * (1.0 + 1e-12):
        L, info = torch.linalg.cholesky_ex(K + j * eye)
        if int(info.max().item()) == 0:
            return L, j
        j *= float(multiplier)
    raise NumericalStabilityError(
        f"Cholesky failed with numerical jitter <= {max_jitter:g}. "
        "Do not increase the cap without changing/documenting the model."
    )


def mvn_log_prob_zero_mean(resid, K, numerical_cfg):
    L, used = strict_cholesky(
        K,
        numerical_cfg.initial_jitter,
        numerical_cfg.max_jitter,
        numerical_cfg.jitter_multiplier,
    )
    sol = torch.cholesky_solve(resid[:, None], L).squeeze(1)
    quad = torch.dot(resid, sol)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    n = resid.numel()
    return -0.5 * (n * LOG2PI + logdet + quad), used


def conditional_gaussian(Ctt, Cst, Css, eps_train, numerical_cfg):
    L, used_train = strict_cholesky(
        Ctt, numerical_cfg.initial_jitter, numerical_cfg.max_jitter,
        numerical_cfg.jitter_multiplier,
    )
    a = torch.cholesky_solve(eps_train[:, None], L)
    mean = (Cst @ a).squeeze(1)
    V = torch.linalg.solve_triangular(L, Cst.T, upper=False)
    cond = 0.5 * ((Css - V.T @ V) + (Css - V.T @ V).T)
    Lc, used_cond = strict_cholesky(
        cond, numerical_cfg.initial_jitter, numerical_cfg.max_jitter,
        numerical_cfg.jitter_multiplier,
    )
    return mean, Lc, max(used_train, used_cond)


class GaussianBNF(nn.Module):
    def __init__(self, mean_input_dim, fit_cfg, prior_cfg, numerical_cfg):
        super().__init__()
        self.mean_net = VariationalMLP(
            mean_input_dim, (fit_cfg.hidden1, fit_cfg.hidden2), 1, "relu"
        )
        self.q_phi = PositiveLogNormalQ(0.35, -2.5)
        self.q_sigma = PositiveLogNormalQ(1.0, -2.5)
        self.prior = prior_cfg
        self.num = numerical_cfg

    def negative_elbo(self, y, x_mean, x_field, beta=1.0):
        mean = self.mean_net(x_mean, sample=True).squeeze(-1)
        phi, lq_phi = self.q_phi.rsample()
        sigma, lq_sigma = self.q_sigma.rsample()
        C = stationary_correlation(x_field, x_field, phi)
        r = (y - mean) / sigma
        ll0, used = mvn_log_prob_zero_mean(r, C, self.num)
        ll = ll0 - y.numel() * torch.log(sigma)
        lp_phi = LogNormal(
            torch.tensor(self.prior.log_phi_mean), torch.tensor(self.prior.log_phi_sd)
        ).log_prob(phi)
        lp_sigma = LogNormal(torch.tensor(0.0), torch.tensor(0.50)).log_prob(sigma)
        kl = self.mean_net.kl(self.prior.mean_neural_sd) + lq_phi + lq_sigma - lp_phi - lp_sigma
        return -(ll - beta * kl), {
            "loglik": float(ll.detach()), "phi": float(phi.detach()),
            "sigma": float(sigma.detach()), "jitter": float(used),
        }

    @torch.no_grad()
    def predictive_samples(self, y_train, x_mean_train, x_field_train,
                           x_mean_test, x_field_test, n_draws=500):
        out, jitters = [], []
        # Same BNN draw for train and test within each posterior draw.
        xm_all = torch.cat([x_mean_train, x_mean_test], dim=0)
        ntr = len(x_mean_train)
        for _ in range(n_draws):
            m_all = self.mean_net(xm_all, sample=True).squeeze(-1)
            mtr, mte = m_all[:ntr], m_all[ntr:]
            phi, _ = self.q_phi.rsample()
            sigma, _ = self.q_sigma.rsample()
            Ctt = stationary_correlation(x_field_train, x_field_train, phi)
            Cst = stationary_correlation(x_field_test, x_field_train, phi)
            Css = stationary_correlation(x_field_test, x_field_test, phi)
            eps_train = (y_train - mtr) / sigma
            eps_mean, Lc, used = conditional_gaussian(Ctt, Cst, Css, eps_train, self.num)
            eps_test = eps_mean + Lc @ torch.randn(len(x_field_test))
            out.append((mte + sigma * eps_test).cpu().numpy())
            jitters.append(used)
        return np.asarray(out), float(max(jitters))


class AdaptiveNSSTBNF(nn.Module):
    def __init__(self, mean_input_dim, field_input_dim, fit_cfg, prior_cfg, numerical_cfg):
        super().__init__()
        h = (fit_cfg.hidden1, fit_cfg.hidden2)
        ha = (fit_cfg.auxiliary_hidden1, fit_cfg.auxiliary_hidden2)
        self.mean_net = VariationalMLP(mean_input_dim, h, 1, "relu")
        self.alpha_net = VariationalMLP(field_input_dim, ha, 1, "tanh")
        self.phi_net = VariationalMLP(field_input_dim, ha, 1, "tanh")
        self.q_alpha0 = NormalScalarQ(0.0, 0.20)
        self.q_logphi0 = NormalScalarQ(math.log(0.35), 0.15)
        self.q_alpha_amp = PositiveLogNormalQ(0.6, -2.5)
        self.q_phi_amp = PositiveLogNormalQ(0.35, -2.5)
        self.q_nutilde = PositiveGammaQ(4.0, 2.0)
        self.q_omega = PositiveGammaQ(30.0, 30.0)
        self.q_u = HalfNormalQ(1.0)
        self.q_sigma = PositiveLogNormalQ(1.0, -2.5)
        self.prior = prior_cfg
        self.num = numerical_cfg

    def _sample_field_scalars(self):
        a0, lq_a0 = self.q_alpha0.rsample()
        lp0, lq_lp0 = self.q_logphi0.rsample()
        aa, lq_aa = self.q_alpha_amp.rsample()
        pa, lq_pa = self.q_phi_amp.rsample()
        return (a0, lp0, aa, pa), (lq_a0, lq_lp0, lq_aa, lq_pa)

    def _fields_from_raw(self, g, h, center_n, scalars):
        a0, lp0, aa, pa = scalars
        # The reference is training-only. For prediction g/h are evaluated on
        # train+test with one network draw, then centered using train entries.
        g = g - g[:center_n].mean()
        h = h - h[:center_n].mean()
        alpha = a0 + aa * g
        delta = alpha / torch.sqrt(1.0 + alpha**2)
        phi = torch.exp(lp0 + pa * h).clamp(0.03, 3.0)
        return alpha, delta, phi

    def fields(self, x_field, sample=True, center_n=None, scalars=None):
        if center_n is None:
            center_n = len(x_field)
        if scalars is None:
            if sample:
                scalars, _ = self._sample_field_scalars()
            else:
                scalars = (
                    self.q_alpha0.mu,
                    self.q_logphi0.mu,
                    torch.exp(self.q_alpha_amp.log_mean),
                    torch.exp(self.q_phi_amp.log_mean),
                )
        g = self.alpha_net(x_field, sample=sample).squeeze(-1)
        h = self.phi_net(x_field, sample=sample).squeeze(-1)
        return self._fields_from_raw(g, h, center_n, scalars)

    def negative_elbo(self, y, x_mean, x_field, beta=1.0):
        mean = self.mean_net(x_mean, sample=True).squeeze(-1)
        scalars, lqs = self._sample_field_scalars()
        alpha, delta, phi = self.fields(
            x_field, sample=True, center_n=len(x_field), scalars=scalars
        )
        nutilde, lq_nu = self.q_nutilde.rsample()
        nu = 2.0 + nutilde
        omega, lq_omega = self.q_omega.rsample()
        u_raw, lq_u = self.q_u.rsample()
        u = u_raw - HALF_NORMAL_MEAN
        sigma, lq_sigma = self.q_sigma.rsample()

        C = adaptive_correlation(x_field, x_field, phi, phi)
        r = torch.sqrt(omega) * (y - mean) / sigma - delta * u
        ll0, used = mvn_log_prob_zero_mean(r, C, self.num)
        ll = ll0 + 0.5 * y.numel() * torch.log(omega) - y.numel() * torch.log(sigma)

        a0, lp0, aa, pa = scalars
        lq_a0, lq_lp0, lq_aa, lq_pa = lqs
        lp_a0 = Normal(torch.tensor(0.0), torch.tensor(1.5)).log_prob(a0)
        lp_lp0 = Normal(torch.tensor(math.log(0.35)), torch.tensor(0.75)).log_prob(lp0)
        lp_aa = LogNormal(torch.tensor(math.log(0.5)), torch.tensor(0.60)).log_prob(aa)
        lp_pa = LogNormal(torch.tensor(math.log(0.30)), torch.tensor(0.60)).log_prob(pa)
        lp_nu = Gamma(torch.tensor(self.prior.nu_shape), torch.tensor(self.prior.nu_rate)).log_prob(nutilde)
        lp_omega = Gamma(nu / 2.0, nu / 2.0).log_prob(omega)
        lp_u = HalfNormal(torch.tensor(1.0)).log_prob(u_raw)
        lp_sigma = LogNormal(torch.tensor(0.0), torch.tensor(0.50)).log_prob(sigma)

        kl = (
            self.mean_net.kl(self.prior.mean_neural_sd)
            + self.alpha_net.kl(self.prior.alpha_neural_sd)
            + self.phi_net.kl(self.prior.range_neural_sd)
            + lq_a0 + lq_lp0 + lq_aa + lq_pa + lq_nu + lq_omega + lq_u + lq_sigma
            - lp_a0 - lp_lp0 - lp_aa - lp_pa - lp_nu - lp_omega - lp_u - lp_sigma
        )
        return -(ll - beta * kl), {
            "loglik": float(ll.detach()), "nu": float(nu.detach()),
            "omega": float(omega.detach()), "u": float(u.detach()),
            "sigma": float(sigma.detach()), "alpha_mean": float(alpha.mean().detach()),
            "phi_mean": float(phi.mean().detach()), "jitter": float(used),
        }

    @torch.no_grad()
    def predictive_samples(self, y_train, x_mean_train, x_field_train,
                           x_mean_test, x_field_test, n_draws=500):
        out, jitters = [], []
        xm_all = torch.cat([x_mean_train, x_mean_test], dim=0)
        xf_all = torch.cat([x_field_train, x_field_test], dim=0)
        ntr = len(x_mean_train)
        for _ in range(n_draws):
            # One posterior network draw is shared by train and test.
            m_all = self.mean_net(xm_all, sample=True).squeeze(-1)
            mtr, mte = m_all[:ntr], m_all[ntr:]
            scalars, _ = self._sample_field_scalars()
            g = self.alpha_net(xf_all, sample=True).squeeze(-1)
            h = self.phi_net(xf_all, sample=True).squeeze(-1)
            alpha, delta, phi = self._fields_from_raw(g, h, ntr, scalars)
            dtr, dte = delta[:ntr], delta[ntr:]
            ptr, pte = phi[:ntr], phi[ntr:]
            nutilde, _ = self.q_nutilde.rsample()
            nu = 2.0 + nutilde
            omega, _ = self.q_omega.rsample()
            u_raw, _ = self.q_u.rsample()
            u = u_raw - HALF_NORMAL_MEAN
            sigma, _ = self.q_sigma.rsample()

            Ctt = adaptive_correlation(x_field_train, x_field_train, ptr, ptr)
            Cst = adaptive_correlation(x_field_test, x_field_train, pte, ptr)
            Css = adaptive_correlation(x_field_test, x_field_test, pte, pte)
            eps_train = torch.sqrt(omega) * (y_train - mtr) / sigma - dtr * u
            eps_mean, Lc, used = conditional_gaussian(Ctt, Cst, Css, eps_train, self.num)
            eps_test = eps_mean + Lc @ torch.randn(len(x_field_test))
            out.append((mte + sigma * (dte * u + eps_test) / torch.sqrt(omega)).cpu().numpy())
            jitters.append(used)
        return np.asarray(out), float(max(jitters))

    @torch.no_grad()
    def surface_summary(self, x_field_ref, x_field_query, n_draws=100):
        xf = torch.cat([x_field_ref, x_field_query], dim=0)
        nref = len(x_field_ref)
        A, P = [], []
        for _ in range(n_draws):
            scalars, _ = self._sample_field_scalars()
            g = self.alpha_net(xf, sample=True).squeeze(-1)
            h = self.phi_net(xf, sample=True).squeeze(-1)
            a, _, p = self._fields_from_raw(g, h, nref, scalars)
            A.append(a[nref:].cpu().numpy())
            P.append(p[nref:].cpu().numpy())
        return np.mean(A, axis=0), np.mean(P, axis=0)


def pretrain_mean_network(mean_net, y_train, x_mean_train, steps=500, lr=8e-3):
    if steps <= 0:
        return
    params = []
    for layer in (mean_net.l1, mean_net.l2, mean_net.l3):
        params += [layer.w_mu, layer.b_mu]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        pred = mean_net(x_mean_train, sample=False).squeeze(-1)
        loss = torch.mean((y_train - pred) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()


def _set_requires(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def initialize_residual_scale(model, y_train, x_mean_train):
    with torch.no_grad():
        resid = y_train - model.mean_net(x_mean_train, sample=False).squeeze(-1)
        s = max(float(torch.std(resid, unbiased=False).cpu()), 0.25)
        model.q_sigma.log_mean.data.fill_(math.log(s))
    return s


def fit_model(model, y_train, x_mean_train, x_field_train, fit_cfg, seed, adaptive=False):
    set_all_seeds(seed)
    pretrain_mean_network(
        model.mean_net, y_train, x_mean_train,
        fit_cfg.mean_pretrain_steps, fit_cfg.mean_pretrain_lr,
    )
    initialize_residual_scale(model, y_train, x_mean_train)
    trace = []

    if adaptive:
        _set_requires(model.mean_net, False)
        pars = [p for p in model.parameters() if p.requires_grad]
        n_stage = max(50, int(fit_cfg.adaptive_stage_fraction * fit_cfg.iterations_adaptive))
        opt = torch.optim.Adam(pars, lr=fit_cfg.learning_rate)
        for it in range(1, n_stage + 1):
            beta = min(1.0, it / max(1, int(0.25 * n_stage)))
            opt.zero_grad(set_to_none=True)
            loss, info = model.negative_elbo(y_train, x_mean_train, x_field_train, beta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite adaptive loss at iteration {it}")
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(pars, fit_cfg.grad_clip))
            opt.step()
            trace.append({"stage": "residual", "iteration": it,
                          "negative_elbo": float(loss.detach()), "grad_norm": gn,
                          "kl_beta": beta, **info})
        _set_requires(model.mean_net, True)
        n_joint = fit_cfg.iterations_adaptive - n_stage
        opt = torch.optim.Adam(model.parameters(), lr=fit_cfg.learning_rate * fit_cfg.adaptive_joint_lr_factor)
        for j in range(1, n_joint + 1):
            opt.zero_grad(set_to_none=True)
            loss, info = model.negative_elbo(y_train, x_mean_train, x_field_train, 1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite adaptive loss at joint iteration {j}")
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), fit_cfg.grad_clip))
            opt.step()
            trace.append({"stage": "joint", "iteration": n_stage + j,
                          "negative_elbo": float(loss.detach()), "grad_norm": gn,
                          "kl_beta": 1.0, **info})
    else:
        opt = torch.optim.Adam(model.parameters(), lr=fit_cfg.learning_rate)
        warm = max(1, int(fit_cfg.kl_warmup_fraction * fit_cfg.iterations_gaussian))
        for it in range(1, fit_cfg.iterations_gaussian + 1):
            beta = min(1.0, it / warm)
            opt.zero_grad(set_to_none=True)
            loss, info = model.negative_elbo(y_train, x_mean_train, x_field_train, beta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Gaussian loss at iteration {it}")
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), fit_cfg.grad_clip))
            opt.step()
            trace.append({"stage": "joint", "iteration": it,
                          "negative_elbo": float(loss.detach()), "grad_norm": gn,
                          "kl_beta": beta, **info})
    return trace


def objective_from_trace(trace, tail=60):
    z = [r["negative_elbo"] for r in trace if r.get("kl_beta", 1.0) >= 0.999999]
    if not z:
        z = [r["negative_elbo"] for r in trace]
    z = z[-min(int(tail), len(z)):]
    return float(np.mean(z))


def sample_crps(y, samples):
    y = np.asarray(y)
    s = np.asarray(samples)
    term1 = np.mean(np.abs(s - y[None, :]), axis=0)
    ss = np.sort(s, axis=0)
    M = s.shape[0]
    i = np.arange(1, M + 1)[:, None]
    pair = (2.0 / (M * M)) * np.sum((2 * i - M - 1) * ss, axis=0)
    return float(np.mean(term1 - 0.5 * pair))


def predictive_metrics(y, samples, nominal=0.95):
    y = np.asarray(y)
    samples = np.asarray(samples)
    pred = samples.mean(axis=0)
    a = 1.0 - nominal
    lo, hi = np.quantile(samples, [a / 2.0, 1.0 - a / 2.0], axis=0)
    return {
        "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
        "MAE": float(np.mean(np.abs(y - pred))),
        "CRPS": sample_crps(y, samples),
        "Coverage": float(np.mean((y >= lo) & (y <= hi))),
        "AIW": float(np.mean(hi - lo)),
    }


def rescale_predictive_dispersion(samples, multiplier):
    samples = np.asarray(samples)
    mu = samples.mean(axis=0, keepdims=True)
    return mu + float(multiplier) * (samples - mu)


def select_calibration_multiplier(y_cal, draws_cal, nominal=0.95,
                                  lo=0.5, hi=3.0, step=0.05):
    rows = []
    for m in np.round(np.arange(lo, hi + 0.5 * step, step), 10):
        met = predictive_metrics(y_cal, rescale_predictive_dispersion(draws_cal, m), nominal)
        rows.append({"multiplier": float(m), **met})
    # Properly allow both shrinkage (m<1) and expansion (m>1). Among candidates
    # reaching nominal coverage choose the sharpest; CRPS breaks ties.
    eligible = [r for r in rows if r["Coverage"] >= nominal]
    if eligible:
        chosen = min(eligible, key=lambda r: (r["AIW"], r["CRPS"], abs(r["Coverage"] - nominal)))
    else:
        chosen = min(rows, key=lambda r: (-r["Coverage"], r["CRPS"], r["AIW"]))
    return float(chosen["multiplier"]), rows

