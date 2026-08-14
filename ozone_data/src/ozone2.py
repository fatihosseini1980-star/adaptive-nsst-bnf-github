"""Deterministic ozone2 application pipeline for the two released models."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from core import (
    GaussianBNF, AdaptiveNSSTBNF, set_all_seeds, to_tensor, fit_model,
    objective_from_trace, predictive_metrics, select_calibration_multiplier,
    rescale_predictive_dispersion, pretrain_mean_network,
)


def _sample_skew(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return 0.0
    s = x.std(ddof=1)
    if s <= 1e-12:
        return 0.0
    z = (x - x.mean()) / s
    return float(n / ((n - 1) * (n - 2)) * np.sum(z ** 3))


def load_ozone2_csv(path):
    df = pd.read_csv(path)
    required = {"station", "day", "longitude", "latitude", "ozone"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.duplicated(["station", "day"]).any():
        raise ValueError("Duplicate station-day rows detected")
    return df.copy()


def station_qc(df, cfg):
    g = df.groupby("station").agg(
        n=("ozone", "count"),
        zeros=("ozone", lambda x: int((x == 0).sum())),
        longitude=("longitude", "first"),
        latitude=("latitude", "first"),
    )
    total_days = df["day"].nunique()
    g["observed_fraction"] = g["n"] / total_days
    g["zero_fraction"] = g["zeros"] / g["n"].replace(0, np.nan)
    g["eligible"] = (
        (g["observed_fraction"] >= cfg.observed_fraction_min)
        & (g["zero_fraction"].fillna(1.0) <= cfg.zero_fraction_max)
    )
    return g.reset_index()


def farthest_point_order(coords, ids, n_select):
    coords = np.asarray(coords, float)
    ids = np.asarray(ids)
    lo = coords.min(axis=0); rr = np.ptp(coords, axis=0); rr[rr == 0] = 1.0
    z = (coords - lo) / rr
    # deterministic first point: western-most, then southern-most tie-break
    first = np.lexsort((z[:, 1], z[:, 0]))[0]
    chosen = [int(first)]
    dist = np.sum((z - z[first])**2, axis=1)
    while len(chosen) < n_select:
        # deterministic tie-break by station id
        best_dist = dist.max()
        candidates = np.where(np.isclose(dist, best_dist, rtol=0, atol=1e-14))[0]
        nxt = int(candidates[np.argmin(ids[candidates])])
        chosen.append(nxt)
        dnew = np.sum((z - z[nxt])**2, axis=1)
        dist = np.minimum(dist, dnew)
        dist[chosen] = -1.0
    return ids[chosen].tolist()


def make_design(df, ozone_cfg):
    qc = station_qc(df, ozone_cfg)
    elig = qc[qc.eligible].sort_values("station")
    if len(elig) < ozone_cfg.panel_size:
        raise ValueError("Not enough QC-eligible stations")
    panel = farthest_point_order(
        elig[["longitude", "latitude"]].to_numpy(),
        elig["station"].to_numpy(), ozone_cfg.panel_size,
    )
    # Purely geometry/order based allocation. No ozone values enter allocation.
    test_pos = np.linspace(0, ozone_cfg.panel_size - 1, ozone_cfg.n_test_stations, dtype=int)
    remaining_pos = [i for i in range(ozone_cfg.panel_size) if i not in set(test_pos)]
    cal_pick = np.linspace(0, len(remaining_pos) - 1, ozone_cfg.n_calibration_stations, dtype=int)
    cal_pos = [remaining_pos[i] for i in cal_pick]
    test_st = [panel[i] for i in test_pos]
    cal_st = [panel[i] for i in cal_pos]
    fit_st = [s for i, s in enumerate(panel) if i not in set(test_pos) and i not in set(cal_pos)]
    days = np.unique(np.rint(np.linspace(df.day.min(), df.day.max(), ozone_cfg.n_days)).astype(int)).tolist()
    return {
        "panel": panel, "fit_stations": fit_st, "calibration_stations": cal_st,
        "test_stations": test_st, "days": days, "qc": qc,
    }


def prepare_arrays(df, design):
    use_st = set(design["fit_stations"] + design["calibration_stations"] + design["test_stations"])
    sub = df[df.station.isin(use_st) & df.day.isin(design["days"]) & df.ozone.notna()].copy()
    fit = sub[sub.station.isin(design["fit_stations"])].copy()
    cal = sub[sub.station.isin(design["calibration_stations"])].copy()
    test = sub[sub.station.isin(design["test_stations"])].copy()
    cols = ["longitude", "latitude", "day"]
    mins = fit[cols].min(); rr = (fit[cols].max() - mins).replace(0, 1)

    # Balance the space-time geometry using training/design coordinates only.
    # The paper defines scaled coordinates u=(s1/a_s,s2/a_s,t/a_t); without
    # this balancing the selected ozone days are about three times denser in
    # normalized time than the stations are in normalized space, producing an
    # unnecessarily ill-conditioned correlation matrix.
    fit_station = fit.drop_duplicates("station")
    zs = ((fit_station[["longitude", "latitude"]] - mins[["longitude", "latitude"]])
          / rr[["longitude", "latitude"]]).to_numpy(float)
    if len(zs) > 1:
        Ds = np.sqrt(np.sum((zs[:, None, :] - zs[None, :, :])**2, axis=2))
        Ds[Ds == 0] = np.nan
        spatial_nn = float(np.nanmedian(np.nanmin(Ds, axis=1)))
    else:
        spatial_nn = 1.0
    td = np.sort(fit["day"].unique())
    tz = (td - mins["day"]) / rr["day"]
    temporal_nn = float(np.median(np.diff(tz))) if len(tz) > 1 else 1.0
    temporal_weight = float(np.clip(spatial_nn / max(temporal_nn, 1e-8), 0.5, 5.0))

    def X(d):
        z = ((d[cols] - mins) / rr).to_numpy(float)
        t = z[:, 2]
        xm = np.c_[z, np.sin(2*np.pi*t), np.cos(2*np.pi*t)]
        xf = z.copy(); xf[:, 2] *= temporal_weight
        return xm, xf

    xmfit, xffit = X(fit); xmcal, xfcal = X(cal); xmtest, xftest = X(test)
    yfit = fit.ozone.to_numpy(float); ycal = cal.ozone.to_numpy(float); ytest = test.ozone.to_numpy(float)
    ym = float(yfit.mean()); ys = float(yfit.std(ddof=0))
    if ys <= 0: raise ValueError("Training response has zero variance")
    yz = (yfit - ym) / ys
    return {
        "fit_df": fit, "cal_df": cal, "test_df": test,
        "xmfit": xmfit, "xffit": xffit, "xmcal": xmcal, "xfcal": xfcal,
        "xmtest": xmtest, "xftest": xftest,
        "yfit": yfit, "ycal": ycal, "ytest": ytest,
        "ym": ym, "ys": ys, "yz": yz,
        "scaling_min": mins, "scaling_range": rr,
        "temporal_weight": temporal_weight,
    }


def initialize_adaptive_fields_training_only(model, y_train_t, xm_train_t, xf_train_np):
    # Initialization only; no calibration/test responses are consulted.
    with torch.no_grad():
        resid = (y_train_t - model.mean_net(xm_train_t, sample=False).squeeze(-1)).cpu().numpy()
    n = len(resid); k = min(45, max(5, n - 1))
    D = np.sum((xf_train_np[:, None, :] - xf_train_np[None, :, :])**2, axis=2)
    inds = np.argsort(D, axis=1)[:, :k]
    targ = np.array([_sample_skew(resid[ii]) for ii in inds])
    targ = np.clip(np.nan_to_num(targ), -1.5, 1.5)
    a0 = float(np.median(targ)); amp = max(float(np.std(targ)), 0.25)
    gtarget = (targ - a0) / amp
    model.q_alpha0.mu.data.fill_(a0)
    model.q_alpha_amp.log_mean.data.fill_(math.log(amp))
    pars = []
    for layer in (model.alpha_net.l1, model.alpha_net.l2, model.alpha_net.l3):
        pars += [layer.w_mu, layer.b_mu]
    opt = torch.optim.Adam(pars, lr=0.01)
    target = to_tensor(gtarget)
    x = to_tensor(xf_train_np)
    for _ in range(350):
        opt.zero_grad(set_to_none=True)
        raw = model.alpha_net(x, sample=False).squeeze(-1)
        raw = raw - raw.mean()
        loss = torch.mean((raw - target)**2) + 1e-3 * sum((p*p).sum() for p in pars)
        loss.backward(); opt.step()
    model.phi_net.set_output_neutral(0.0)
    model.q_phi_amp.log_mean.data.fill_(math.log(0.15))
    model.q_logphi0.mu.data.fill_(math.log(0.30))
    return {"alpha_init_median": a0, "alpha_init_sd": float(np.std(targ))}


def _run_restarts(model_name, arrays, exp_cfg):
    import copy
    fit_cfg, prior, num = exp_cfg.fit, exp_cfg.prior, exp_cfg.numerical
    xmfit = to_tensor(arrays["xmfit"]); xffit = to_tensor(arrays["xffit"])
    yfit_t = to_tensor(arrays["yz"])
    rows, states, traces = [], {}, {}

    def new_model():
        if model_name == "Gaussian BNF":
            return GaussianBNF(arrays["xmfit"].shape[1], fit_cfg, prior, num), False
        if model_name == "Adaptive NSST-BNF":
            return AdaptiveNSSTBNF(arrays["xmfit"].shape[1], arrays["xffit"].shape[1], fit_cfg, prior, num), True
        raise ValueError(model_name)

    for seed in fit_cfg.restart_seeds:
        set_all_seeds(seed)  # before model construction
        model, adaptive = new_model()
        init = {}
        local_fit_cfg = copy.deepcopy(fit_cfg)
        if adaptive:
            pretrain_mean_network(model.mean_net, yfit_t, xmfit,
                                  local_fit_cfg.mean_pretrain_steps, local_fit_cfg.mean_pretrain_lr)
            init = initialize_adaptive_fields_training_only(model, yfit_t, xmfit, arrays["xffit"])
            local_fit_cfg.mean_pretrain_steps = 0
        trace = fit_model(model, yfit_t, xmfit, xffit, local_fit_cfg, seed, adaptive=adaptive)
        obj = objective_from_trace(trace, fit_cfg.objective_tail)
        row = {"model": model_name, "seed": int(seed), "objective": obj,
               "max_fit_jitter": max(float(r.get("jitter", 0)) for r in trace), **init}
        rows.append(row)
        # Store only parameter tensors and traces, not three live model graphs.
        states[int(seed)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        traces[int(seed)] = trace
        del model

    table = pd.DataFrame(rows).sort_values(["objective", "seed"]).reset_index(drop=True)
    best_seed = int(table.iloc[0].seed)
    model, _ = new_model()
    model.load_state_dict(states[best_seed])
    return model, best_seed, table, traces[best_seed]

def evaluate_model(model_name, arrays, exp_cfg):
    model, seed, restarts, trace = _run_restarts(model_name, arrays, exp_cfg)
    T = to_tensor
    ytr = T(arrays["yz"])
    xmtr, xftr = T(arrays["xmfit"]), T(arrays["xffit"])
    xmcal, xfcal = T(arrays["xmcal"]), T(arrays["xfcal"])
    xmte, xfte = T(arrays["xmtest"]), T(arrays["xftest"])

    cal_std, cal_jit = model.predictive_samples(
        ytr, xmtr, xftr, xmcal, xfcal, exp_cfg.fit.posterior_draws
    )
    cal_draws = cal_std * arrays["ys"] + arrays["ym"]
    mult, grid = select_calibration_multiplier(
        arrays["ycal"], cal_draws, exp_cfg.ozone.nominal_coverage,
        exp_cfg.ozone.calibration_min, exp_cfg.ozone.calibration_max,
        exp_cfg.ozone.calibration_step,
    )
    test_std, test_jit = model.predictive_samples(
        ytr, xmtr, xftr, xmte, xfte, exp_cfg.fit.posterior_draws
    )
    test_raw = test_std * arrays["ys"] + arrays["ym"]
    test_cal = rescale_predictive_dispersion(test_raw, mult)
    raw = predictive_metrics(arrays["ytest"], test_raw, exp_cfg.ozone.nominal_coverage)
    calibrated = predictive_metrics(arrays["ytest"], test_cal, exp_cfg.ozone.nominal_coverage)
    summary = {
        "Model": model_name, "best_seed": seed, "multiplier": mult,
        "n_fit": len(arrays["yfit"]), "n_cal": len(arrays["ycal"]), "n_test": len(arrays["ytest"]),
        "max_fit_jitter": float(restarts.max_fit_jitter.max()),
        "max_calibration_prediction_jitter": cal_jit,
        "max_test_prediction_jitter": test_jit,
        **{f"Raw_{k}": v for k, v in raw.items()},
        **{f"Cal_{k}": v for k, v in calibrated.items()},
    }
    if model_name == "Adaptive NSST-BNF":
        query = np.vstack([arrays["xffit"], arrays["xfcal"], arrays["xftest"]])
        a, p = model.surface_summary(xftr, T(query), exp_cfg.fit.surface_draws)
        summary.update({
            "AlphaMean": float(np.mean(a)), "AlphaSD": float(np.std(a)),
            "AlphaMin": float(np.min(a)), "AlphaMax": float(np.max(a)),
            "PhiMean": float(np.mean(p)), "PhiSD": float(np.std(p)),
            "PhiMin": float(np.min(p)), "PhiMax": float(np.max(p)),
        })
    pred = pd.DataFrame({
        "station": arrays["test_df"].station.to_numpy(int),
        "day": arrays["test_df"].day.to_numpy(int),
        "longitude": arrays["test_df"].longitude.to_numpy(float),
        "latitude": arrays["test_df"].latitude.to_numpy(float),
        "observed": arrays["ytest"],
        "pred_mean": test_cal.mean(axis=0),
        "lo95": np.quantile(test_cal, .025, axis=0),
        "hi95": np.quantile(test_cal, .975, axis=0),
    })
    return summary, restarts, pd.DataFrame(grid), pred, trace
