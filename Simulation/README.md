# Adaptive NSST-BNF Simulation Study

Reproducibility repository for the simulation study of **Adaptive Skew-t Bayesian Neural Fields for Spatio-Temporal Prediction**.

This repository contains only the simulation code and simulation outputs. It does **not** contain the manuscript, reviewer-response files, or the real-data Ozone application.

## What is included

Four methods are evaluated on the same generated datasets and the same train/test split within every replication:

1. **Gaussian Neural Model** — Bayesian neural-network mean with stationary Gaussian residual correlation.
2. **Stationary Skew-t Neural Model** — the same Bayesian neural mean with stationary skew-t residual structure.
3. **Sparse Variational GP (SVGP)** — ARD RBF sparse variational Gaussian process with 24 inducing locations and learned observation noise.
4. **Proposed Adaptive NSST-BNF** — Bayesian neural mean with adaptive skewness and local-range neural functions under the nonstationary skew-t residual construction.

## Final simulation design

- Five data-generating scenarios.
- `n = 80` observations per replication.
- 80/20 train/test split.
- 40 independent final replications per scenario.
- Final replication IDs: `1001`--`1040`.
- 300 posterior predictive draws per fitted model in the final Monte Carlo run.
- Proposed-model predictive-dispersion multiplier: **1.20**.
- The value 1.20 is selected **before the final Monte Carlo experiment** by minimizing mean CRPS over 25 independent development fits: 5 scenarios × replications `2001`--`2005`.
- Prior-sensitivity replication IDs: `3001`--`3005`, disjoint from both the development and final Monte Carlo runs.

### Five scenarios

The nonlinear mean function is shared across scenarios. The residual mechanism becomes progressively more demanding:

- **Scenario 1:** stationary Gaussian residuals, zero skewness, stationary range.
- **Scenario 2:** symmetric heavy-tailed residuals, zero skewness, stationary range.
- **Scenario 3:** stationary skewed heavy-tailed residuals with constant skewness and stationary range.
- **Scenario 4:** adaptive skewness and nonstationary local range.
- **Scenario 5:** stronger adaptive skewness and stronger local-range variation.

The exact data-generating equations are implemented in `src/core.py` (`true_mean`, `true_alpha`, `true_phi`, and `simulate_dataset`).

## Important implementation details

The final implementation uses the following prediction conventions consistently across the neural models:

- A **single posterior draw of the Bayesian mean network is shared across training and test locations within each predictive draw**.
- For the proposed model, local skewness/range deviations are **centered using training locations only**. Test locations do not affect the centering reference.
- The centered half-normal latent variable is

  `U_c = |N(0,1)| - sqrt(2/pi)`.

- The global residual scale is estimated explicitly.
- The same generated dataset and train/test split are used for every competing method within a replication.
- Calibration of the proposed predictive dispersion is performed only on development replications, never on the final replications.

## Final corrected results

Mean values across 40 replications are shown below. Full mean/SD, median, IQR, and standard-error summaries are in `results/`.

| Scenario | Model | RMSE | MAE | CRPS | Coverage | AIW |
|---|---|---:|---:|---:|---:|---:|
| 1 | Gaussian Neural Model | 0.309 | 0.204 | 0.146 | 0.920 | 0.919 |
| 1 | Stationary Skew-t Neural Model | 0.305 | 0.199 | 0.142 | 0.931 | 0.941 |
| 1 | SVGP | 0.391 | 0.285 | 0.207 | 0.969 | 1.653 |
| 1 | Proposed Adaptive NSST-BNF | **0.279** | **0.177** | **0.127** | 0.966 | 1.075 |
| 2 | Gaussian Neural Model | 0.296 | 0.208 | 0.147 | 0.917 | 0.890 |
| 2 | Stationary Skew-t Neural Model | 0.294 | 0.207 | 0.146 | 0.931 | 0.904 |
| 2 | SVGP | 0.371 | 0.277 | 0.205 | 0.977 | 1.745 |
| 2 | Proposed Adaptive NSST-BNF | **0.263** | **0.179** | **0.125** | 0.978 | 1.041 |
| 3 | Gaussian Neural Model | 0.310 | 0.216 | 0.153 | 0.894 | 0.897 |
| 3 | Stationary Skew-t Neural Model | 0.307 | 0.212 | 0.149 | 0.912 | 0.924 |
| 3 | SVGP | 0.383 | 0.294 | 0.216 | 0.980 | 1.796 |
| 3 | Proposed Adaptive NSST-BNF | **0.273** | **0.187** | **0.129** | 0.961 | 1.086 |
| 4 | Gaussian Neural Model | 0.529 | 0.393 | 0.300 | 0.791 | 1.137 |
| 4 | Stationary Skew-t Neural Model | 0.524 | 0.389 | 0.299 | 0.786 | 1.145 |
| 4 | SVGP | 0.497 | 0.378 | 0.275 | 0.977 | 2.234 |
| 4 | Proposed Adaptive NSST-BNF | **0.480** | **0.354** | **0.257** | 0.925 | 1.593 |
| 5 | Gaussian Neural Model | 0.463 | 0.341 | 0.255 | 0.794 | 1.044 |
| 5 | Stationary Skew-t Neural Model | 0.468 | 0.344 | 0.257 | 0.792 | 1.062 |
| 5 | SVGP | **0.436** | 0.334 | 0.238 | 0.983 | 1.956 |
| 5 | Proposed Adaptive NSST-BNF | 0.441 | **0.319** | **0.229** | 0.900 | **1.432** |

The proposed method is not presented as uniformly best on every metric. In Scenario 5, for example, SVGP has slightly lower mean RMSE and coverage closer to 0.95, whereas the proposed method has lower mean MAE and CRPS and a narrower average prediction interval.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── src/
│   ├── config.py
│   ├── core.py
│   ├── svgp.py
│   ├── run_simulation.py
│   ├── run_dispersion_calibration.py
│   ├── run_final_40rep.py
│   └── run_prior_sensitivity_final.py
├── scripts/
│   ├── summarize_results.py
│   └── make_figures.py
├── tests/
│   └── test_core.py
├── results/
│   ├── final_raw_results_40rep.csv
│   ├── final_summary_mean_sd_se_median_iqr.csv
│   ├── final_table_mean_sd.csv
│   ├── development_calibration_raw.csv
│   ├── development_calibration_summary.csv
│   ├── prior_sensitivity_raw_5rep.csv
│   ├── prior_sensitivity_summary_5rep.csv
│   ├── convergence_summary.csv
│   ├── surface_recovery_summary.csv
│   └── paired_differences_proposed_minus_benchmark.csv
└── figures/
    ├── crps_final.png
    └── coverage_final.png
```

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick smoke test

```bash
python src/run_simulation.py --smoke --out outputs/smoke
```

This executes a deliberately small run to verify that all four model paths work.

Run unit tests with:

```bash
python -m pytest -q
```

## Reproduce the development calibration

```bash
python src/run_dispersion_calibration.py --out outputs/development_calibration
```

The script evaluates the frozen factor grid and writes the selected value to `chosen_factor.txt`. With the reported design the selected multiplier is 1.20.

## Reproduce the final 40-replication simulation

```bash
python src/run_final_40rep.py
```

This is computationally intensive. The driver is resumable: completed Scenario × Replication × Model combinations are retained in the output CSV.

## Reproduce prior sensitivity

```bash
python src/run_prior_sensitivity_final.py
```

The analysis uses strong/baseline/weak neural-prior standard deviations equal to 0.5×/1×/2× the baseline values for Scenarios 4 and 5.

## Rebuild summaries and figures

```bash
python scripts/summarize_results.py
python scripts/make_figures.py
```

## Notes on latent-surface recovery

The repository reports quantitative recovery metrics for the adaptive skewness and local-range surfaces in Scenarios 4 and 5. These diagnostics should be interpreted separately from predictive performance: component-wise latent-surface recovery is substantially more difficult than out-of-sample prediction because the flexible mean, skewness, and dependence components can be weakly identified. The relevant metrics are retained in `results/surface_recovery_summary.csv` rather than being inferred from visual inspection alone.

## Citation

Citation metadata are provided in `CITATION.cff`.
