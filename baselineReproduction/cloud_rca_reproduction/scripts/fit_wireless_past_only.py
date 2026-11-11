"""Fit an exploratory past-only, zero-hop Hawkes model to Wireless alarm counts.

NOT the paper's TTHP, and not the paper's EM-based graph search.
No DAG.npy is read here. Train/validation split is chronological.
Run from baselineReproduction/cloud_rca_reproduction with .venv-thp active.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.fit_past_only_thp import objective_and_grad, past_effects


def make_counts(events, n_bins=512):
    if not isinstance(n_bins, int) or not 16 <= n_bins <= 2048:
        raise ValueError("n_bins must be an integer from 16 to 2048")
    columns = {"timestamp", "node", "event"}
    if not columns.issubset(events):
        raise ValueError(f"Expected columns {sorted(columns)}")
    if events.empty or events[list(columns)].isna().any().any():
        raise ValueError("Missing or invalid Wireless events")

    times = events["timestamp"].to_numpy(dtype=float)
    nodes = events["node"].to_numpy(dtype=int)
    types = events["event"].to_numpy(dtype=int)
    if (not np.isfinite(times).all() or np.min(nodes) < 0 or np.min(types) < 0
        or not np.all(events["node"] == nodes) or not np.all(events["event"] == types)):
        raise ValueError("Timestamp/node/event values must be finite, with integer nonnegative IDs")
    if np.unique(nodes).tolist() != list(range(int(nodes.max()) + 1)):
        raise ValueError("Node IDs must be consecutive; use prepare_wireless.py mapping")
    if np.unique(types).tolist() != list(range(int(types.max()) + 1)):
        raise ValueError("Event IDs must be consecutive 0..K-1")
    t_min = float(times.min())
    t_max = float(times.max())
    width = (t_max - t_min + 1.0) / n_bins
    if width <= 0 or not np.isfinite(width):
        raise ValueError("Invalid timestamp range")
    b = np.clip(np.floor((times - t_min) / width).astype(int), 0, n_bins - 1)
    x = np.zeros((int(nodes.max()) + 1, n_bins, int(types.max()) + 1), dtype=np.float64)
    np.add.at(x, (nodes, b, types), 1)
    if int(x.sum()) != len(events):
        raise AssertionError("Binning lost or double-counted some events")
    return x, {"min_timestamp":t_min, "max_timestamp":t_max,
               "bin_width_timestamp_units":width, "bins":n_bins}


def fit(x, maxiter=150, decay_per_bin=0.6, penalty=0.02):
    # Treat past events in prior bins as the only explanatory history.
    hist = past_effects(x, decay=decay_per_bin)
    cutoff = int(x.shape[1] * 0.75)
    y_train, h_train = x[:, :cutoff], hist[:, :cutoff]
    y_valid, h_valid = x[:, cutoff:], hist[:, cutoff:]
    p = x.shape[-1]
    x0 = np.r_[np.maximum(y_train.mean(axis=(0,1)), 1e-6), np.zeros(p*p)]
    result = minimize(objective_and_grad, x0,
        args=(y_train, h_train, penalty), jac=True,
        method="L-BFGS-B", bounds=[(1e-6, None)]*p + [(0,None)]*(p*p),
        options={"maxiter":maxiter, "ftol":1e-9})
    def score(params, yy, hh):
        # Compare *unpenalized* NLL consistently for both models.
        return float(objective_and_grad(params, yy, hh, penalty=0)[0]/yy.size)
    metrics = {
      "converged":bool(result.success), "optimizer_message":str(result.message),
      "optimizer_iterations":int(result.nit), "train_bins":cutoff,
      "validation_bins":int(x.shape[1]-cutoff),
      "train_observations":int(y_train.sum()),
      "validation_observations":int(y_valid.sum()),
      "baseline_train_nll_per_cell":score(x0,y_train,h_train),
      "fitted_train_nll_per_cell":score(result.x,y_train,h_train),
      "baseline_validation_nll_per_cell":score(x0,y_valid,h_valid),
      "fitted_validation_nll_per_cell":score(result.x,y_valid,h_valid),
      "decay_per_bin":decay_per_bin,"l2_alpha_penalty":penalty,
      "maxiter":maxiter,"scipy_objective":"discrete Poisson NLL, same-device, past-only", 
      "not_paper_tthp":True,
      "note":"Validation uses causal prior-bin histories (including observations from the earlier training period); no target current/future bins enter features."
    }
    return result, metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events",default="data/processed/wireless_thp_events.csv")
    parser.add_argument("--bins",type=int,default=512)
    parser.add_argument("--maxiter",type=int,default=150)
    parser.add_argument("--output",default="outputs/wireless_past_only_bin512")
    a=parser.parse_args()
    if a.maxiter < 1:
        parser.error("--maxiter must be >= 1")
    events_path = Path(a.events)
    df = pd.read_csv(events_path)
    x, bin_meta = make_counts(df,a.bins)
    print(f"EVENTS={len(df)}, NODES={x.shape[0]}, TYPES={x.shape[2]}, BINS={a.bins}", flush=True)
    print("Timestamp units are UNKNOWN; decay=0.6 is per bin, not per second", flush=True)
    result, metrics = fit(x,maxiter=a.maxiter)
    metrics.update(bin_meta)
    metrics["source_csv_sha256"] = hashlib.sha256(events_path.read_bytes()).hexdigest()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    p=x.shape[-1]
    np.savez_compressed(out/"parameters.npz", mu=result.x[:p],alpha=result.x[p:].reshape(p,p))
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2),flush=True)
    if not result.success:
        raise SystemExit("Optimizer did NOT converge. Logs and diagnostic parameters saved; no trained-model claim.")
    print("Past-only model fitted; saved",out,flush=True)
    print("NO causal graph was selected; DAG labels were never used",flush=True)


if __name__=="__main__":
    main()
