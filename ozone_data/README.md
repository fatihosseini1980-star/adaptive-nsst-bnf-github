# Ozone Data

Real-data analysis for the `ozone2` spatio-temporal ozone dataset.

This folder is intentionally limited to the real-data application and contains
only two models:

1. **Gaussian BNF**
2. **Adaptive NSST-BNF**

## Contents

```text
ozone_data/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── scripts/
│   └── export_ozone2.R
├── src/
│   ├── config.py
│   ├── core.py
│   ├── ozone2.py
│   ├── run_ozone2.py
│   └── run_ozone2_restart.py
├── tests/
│   └── test_core.py
└── results/
    └── [reference ozone2 outputs]
```

## Data

The application uses `ozone2` from the R package `fields`. Generate the long
CSV used by the Python analysis with:

```bash
Rscript scripts/export_ozone2.R
```

The command creates `data/ozone2_long.csv`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the analysis

From inside the `ozone_data` directory:

```bash
python src/run_ozone2.py --data data/ozone2_long.csv --out results/ozone2_run
```

For a short code check:

```bash
python src/run_ozone2.py --data data/ozone2_long.csv --out results/smoke --quick
```

Each model is fitted from three predeclared restart seeds (`111`, `222`, `333`).
Restart selection uses the terminal training objective only. Predictive
dispersion calibration uses calibration stations only and is separate from the
locked test stations.

## Numerical safeguards

- Both models use the same mean-network features.
- Spatial and temporal coordinates are scaled from the training design only.
- Within each posterior predictive draw, the same sampled neural-network
  weights are used for training and prediction locations.
- Adaptive-field centering is defined using training locations only.
- Cholesky jitter is numerical stabilization only: it starts at `1e-8`, is
  capped at `1e-4`, and the run fails instead of silently using a larger value.
- No hidden nugget term is introduced.
- Random seeds are set before model construction.
- Calibration can either contract or expand predictive dispersion.

## Reference result

The checked reference output is `results/two_model_comparison.csv`.

| Model | RMSE | MAE | Calibrated CRPS | 95% coverage | AIW |
|---|---:|---:|---:|---:|---:|
| Gaussian BNF | 12.881 | 9.450 | 7.225 | 0.989 | 69.140 |
| Adaptive NSST-BNF | **10.508** | **8.272** | **6.311** | **0.979** | **64.668** |

