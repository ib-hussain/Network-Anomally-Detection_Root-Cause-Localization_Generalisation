# Step 21 — Real Wireless parameter-fitting baseline (not TTHP)

This exercise tests whether the **strictly past-only, same-device** parameter estimator
from Step 20 can fit actual Wireless records. It is **not the authors' TTHP model**.

## Prerequisites

From the repository's `baselineReproduction/cloud_rca_reproduction/` directory:

```bash
source .venv-thp/bin/activate
python -m pip check
python -m unittest discover -s tests -v
```

The processed input must exist at `data/processed/wireless_thp_events.csv`. If missing,
rerun `python scripts/prepare_wireless.py` after checking the downloaded raw CSV.

## Run

```bash
mkdir -p outputs
set -o pipefail
python -u -m scripts.fit_wireless_past_only --bins 512 --maxiter 150 \
    --output outputs/wireless_past_only_bin512 2>&1 | tee outputs/step21_wireless.log
```

Inspect `outputs/wireless_past_only_bin512/metrics.json` and `parameters.npz`.
If convergence is false, do **not** claim successful fitting; retain diagnostics.
You may rerun with `--maxiter 300` and **a new output directory**, but do not
select the iteration setting using the ground-truth DAG as an oracle.

## Model and evaluation definition

- Chronological split: first 75% of bins for training, final 25% for validation.
- A fixed grid of 512 equally spaced bins over the full dataset timestamp range;
  `bin_width_timestamp_units` is saved. Source timestamps' physical units are
  not established, so **never call this seconds or milliseconds without evidence**.
- The model is discrete Poisson counts (node, time bin, alarm type). Each bin is
  treated as one unit of exposure. Decay coefficient 0.6 is **per bin**.
- At time bin t, features use bins **strictly before** t, from the same device.
  Validation uses causal history from previous bins, including training-period
  history. It never uses current/future target-bin counts as input.
- Source-to-target alpha is nonnegative; mu is strictly positive.
- L-BFGS-B optimization, L2 alpha penalty 0.02, not the paper's EM algorithm.
- Compare unpenalized negative log-likelihood (NLL per cell) on train/validation
  against a constant-rate Poisson baseline fitted on training counts.
- **A lower validation NLL** is a predictive-fit diagnostic, **not causal accuracy**.
- `alpha` is a dense parameter matrix, **not a selected causal graph**; don't
  threshold it by tuning against `DAG.npy` and call it the published TTHP result.
- Input CSV hash, bin width, optimizer outcome, and full parameter arrays are saved.
- Does not use the ground-truth DAG, device topology, retrospective delay window,
  TTHP penalized graph search, ACG, GAT, or LightGBM.

## Git checkpoint

The `data/` and `outputs/` directories are ignored. Stage **only code, tests and this page**:

```bash
git add scripts/fit_wireless_past_only.py tests/test_wireless_past_only.py setup/STEP21_WIRELESS_PAST_ONLY.md
git diff --cached --stat
git commit -m "exp(cloud-rca): validate past-only Hawkes fitting on Wireless"
git status --short
```
