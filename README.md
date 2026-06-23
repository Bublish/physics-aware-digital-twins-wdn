# Towards Physics-Aware Digital Twins

**Evaluating Explainable Machine Learning for Predictive Control in Water Distribution Networks**

This repository hosts the code and data accompanying the paper
*"Towards Physics-Aware Digital Twins: Evaluating Explainable Machine Learning for
Predictive Control in Water Distribution Networks"* (Kevin Stele and Jens Wala),
submitted to the **37. Forum Bauinformatik 2026, Rostock**.

## Overview

Digital Twins of water distribution networks rely on hydraulic solvers such as EPANET,
which are physically accurate but computationally expensive for real-time control.
Data-driven surrogate models offer a fast alternative, but their black-box nature is a
concern for critical infrastructure. This study compares four machine-learning
architectures as surrogate engines for a hydraulic Digital Twin and uses explainable-AI
methods (SHAP, permutation importance) to assess how *physically grounded* each model's
predictions are.

The study addresses two research questions:

- **RQ1** — Which ML architecture best balances predictive accuracy, efficiency and
  physical interpretability as a Digital-Twin surrogate?
- **RQ2** — Can post-hoc XAI methods (SHAP, permutation importance) reliably distinguish
  physically grounded models from statistically driven ones?

## Models compared

| Model | Library | Type |
|---|---|---|
| XGBoost | `xgboost` | Gradient-boosted trees |
| Multi-Layer Perceptron (MLP) | `scikit-learn` | Feed-forward neural network |
| Generalized Additive Model (GAM) | `pyGAM` | Additive splines |
| Explainable Boosting Machine (EBM) | `interpret` | Glass-box boosted GAM |

All four predict pressure, hydraulic head and demand simultaneously from a 19-dimensional
feature vector (node elevation, base demand, the demand-to-elevation ratio, the hour of
day, and lagged demand/head/pressure features). Hyperparameters are tuned with Bayesian
optimisation (`scikit-optimize`, GP surrogate, cross-validated R²).

## Repository structure

```
.
├── EPANET Analysis_30D.ipynb   # Primary: 30-day dataset (epyt API-seeded), tuning + evaluation
├── EPANET Analysis_4D.ipynb    # Comparison: 96-hour (4-day) report-parsed run, same pipeline
├── data/
│   ├── WSN1.inp                # EPANET input model: Water Sensor Network 1 (WSN1)
│   └── WSN1 - report.txt.gz    # 96-hour EPANET text report (gzipped; parsed by the 4-day notebook)
├── requirements.txt
├── LICENSE                     # MIT
└── README.md
```

### The two notebooks

- **`EPANET Analysis_30D.ipynb` (primary).** Drives EPANET through the `epyt` API to
  generate a 30-day horizon at a 15-minute step. PATTERN-0 is augmented with per-step
  lognormal jitter (σ = 0.15) so successive days vary rather than repeat.
- **`EPANET Analysis_4D.ipynb` (comparison).** Parses the original 96-hour EPANET report
  (`data/WSN1 - report.txt`) at a 0.5-hour step and pushes it through the same downstream
  pipeline, as a small-vs-large robustness comparison.

The study network is **Water Sensor Network 1 (WSN1)** [Ostfeld, *Battle of the Water
Network Models 3*, 2021], a benchmark network with one reservoir, two pump stations, two
tanks and ~23 miles of pipe. Simulations use EPANET 2.2 with the Darcy–Weisbach headloss
formula under demand-driven analysis (DDA).

## Getting started

### 1. Install dependencies

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Decompress the 4-day report

The 4-day EPANET report is stored gzipped (the raw text is ~118 MB, over GitHub's
file-size limit). Decompress it before running the 4-day notebook:

```bash
# macOS/Linux
gunzip -k "data/WSN1 - report.txt.gz"
# Windows PowerShell (requires the report alongside the .gz):
#   Use 7-Zip / tar, or run: tar -xzf "data/WSN1 - report.txt.gz" -C data
```

This produces `data/WSN1 - report.txt`. The 30-day notebook does **not** need this file —
it regenerates its data from `data/WSN1.inp` via the EPANET engine.

### 3. Run the notebooks

```bash
jupyter lab    # or: jupyter notebook
```

> **Data paths.** The notebooks were written against the authors' working tree and look
> for `WSN1.inp` / `WSN1 - report.txt` at a few candidate locations. If a path is not
> found, point the relevant load cell at the files in `data/`.

## Results summary

After Bayesian tuning, all four models reach high accuracy on the pressure target
(R² ≥ 0.997 on the 30-day set). The decisive differences are interpretability and
training cost rather than raw error.

Pressure prediction (RMSE [psi] / combined train + inference time [s]):

| Model | 4-day RMSE | 4-day Time | 30-day RMSE | 30-day Time |
|---|---|---|---|---|
| XGBoost | 0.89 | 5.5 | 0.52 | 7.7 |
| Neural Network (MLP) | 1.87 | 26.9 | 2.94 | 28.7 |
| GAM | 2.91 | 1.4 | 3.70 | 4.0 |
| EBM | 2.16 | 1981 | 3.59 | 934 |

SHAP analysis shows the models reach comparable accuracy through different feature
strategies: XGBoost relies almost entirely on short-horizon autoregressive pressure lags
(statistical correlation), whereas the MLP grounds its predictions in elevation and the
hydrostatic elevation–head–pressure relationship. Physical grounding and global accuracy
are distinct, partially competing properties.

## License

Code and configuration in this repository are released under the [MIT License](LICENSE).
The WSN1 benchmark network is from the *Battle of the Water Network Models 3* (Ostfeld,
2021); please credit the original source when reusing it.
