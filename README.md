# Adaptive Skew-t Bayesian Neural Fields for Spatio-Temporal Prediction

Reproducibility code for the revised manuscript by **Fatemeh Hosseini** and **Omid Karimi**.

## What is included

The repository contains the complete revised simulation pipeline used in the manuscript:

- five spatio-temporal data-generating scenarios;
- Gaussian neural benchmark;
- stationary skew-\(t\) neural benchmark;
- sparse variational Gaussian process (SVGP) benchmark;
- proposed adaptive NSST-BNF;
- 40-replication final Monte Carlo driver;
- posterior predictive evaluation (RMSE, MAE, CRPS, 95% coverage, AIW);
- objective-based convergence diagnostics;
- predictive-dispersion calibration code;
- prior-sensitivity analysis for Scenarios 4 and 5;
- final replication-level results and manuscript-ready summaries;
- scripts for the CRPS and coverage figures.

The latent-surface recovery figures from an earlier development version are **not part of the revised analysis**.

## Final simulation design

- observations per replication: `n = 80`
- train/test split: `80/20`
- scenarios: `5`
- final replications per scenario: `40`
- models: `4`
- final replication IDs: `1001 ... 1040`
- Gaussian and stationary skew-\(t\) neural models: maximum `600` iterations
- SVGP and proposed model: maximum `1000` iterations
- posterior predictive draws: `300`
- proposed-model predictive-dispersion multiplier: `1.50`
- SVGP inducing locations: `24`

The dispersion multiplier was selected using development replications disjoint from the final 40-replication experiment and then fixed before the final rerun.

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce the final Monte Carlo experiment

From the repository root:

```bash
python src/run_final_40rep.py
```

The script is resumable. Results are written to:

```text
outputs/final_40rep/
```

The final published/revised run is also supplied in:

```text
results/final_raw_results_40rep.csv
```

## Reproduce the prior-sensitivity analysis

The manuscript sensitivity check uses strong, baseline, and weak neural priors, corresponding to multipliers of `0.5`, `1.0`, and `2.0` applied to the baseline neural prior standard deviations.

It is evaluated in Scenarios 4 and 5 using five independent replications per prior setting:

```bash
python src/run_prior_sensitivity_final.py
```

Saved reference results are in:

```text
results/prior_sensitivity_raw_5rep.csv
results/prior_sensitivity_summary_5rep.csv
```

## Reproduce the predictive-dispersion calibration

The development-only calibration grid can be rerun with:

```bash
python src/run_dispersion_calibration.py --scenario 5 --reps 10 \
  --rep-start 2001 --iterations 1000 --draws 300 --n-total 80 \
  --out outputs/calibration_s5
```

Development/calibration replications must remain disjoint from final Monte Carlo IDs.

## Rebuild result summaries

```bash
python scripts/summarize_results.py
```

## Rebuild figures

```bash
python scripts/make_figures.py
```

## Repository structure

```text
adaptive-nsst-bnf/
├── README.md
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py
│   ├── core.py
│   ├── svgp.py
│   ├── run_simulation.py
│   ├── run_final_40rep.py
│   ├── run_prior_sensitivity_final.py
│   └── run_dispersion_calibration.py
├── scripts/
│   ├── summarize_results.py
│   └── make_figures.py
├── results/
├── figures/
└── manuscript/
```

## Reproducibility note

All final tables should be regenerated from the saved raw replication-level results rather than manually edited. The code fixes the random-seed construction and uses the same generated dataset and train/test split for all competing models within each replication.

## License

No software license is included in this package. Add the license you want before making the repository public (for example, MIT or BSD-3-Clause if you want to permit broad reuse).
