"""Full Wireless THP without device topology.

Exploratory baseline; NOT the paper's TTHP reproduction.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from castle.algorithms import TTPM

data_path = Path("data/processed/wireless_thp_events.csv")
dag_path = Path(
    "data/raw/wireless/18V_55N_Wireless/DAG.npy"
)
output = Path("outputs/thp_18_no_topology")
output.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(data_path)
truth = np.load(dag_path, allow_pickle=False)

events = sorted(data["event"].unique().tolist())
nodes = sorted(data["node"].unique().tolist())

assert events == list(range(18))
assert nodes == list(range(55))
assert truth.shape == (18, 18)
assert len(data) == 34838

# Identity is a placeholder for an experiment with max_hop=0.
# It is NOT the physical Wireless network topology.
topology = np.eye(55, dtype=int)

print("=== FULL WIRELESS THP ===", flush=True)
print("Events:", len(data), flush=True)
print("Alarm types:", len(events), flush=True)
print("Devices:", len(nodes), flush=True)
print("Topology mode: NO CROSS-DEVICE NEIGHBORS", flush=True)

start = time.monotonic()

model = TTPM(
    topology_matrix=topology,
    max_hop=0,
    max_iter=1,
    delta=0.1,
    penalty="BIC",
)

model.learn(data[["event", "timestamp", "node"]].copy())

elapsed = time.monotonic() - start
pred = np.asarray(model.causal_matrix)

assert pred.shape == truth.shape

np.save(output / "causal_graph.npy", pred)

# Directed, off-diagonal edge evaluation.
true_edges = truth != 0
pred_edges = pred != 0

np.fill_diagonal(true_edges, False)
np.fill_diagonal(pred_edges, False)

tp = int(np.logical_and(pred_edges, true_edges).sum())
fp = int(np.logical_and(pred_edges, ~true_edges).sum())
fn = int(np.logical_and(~pred_edges, true_edges).sum())

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

results = {
    "experiment": "THP-18-NoTopology",
    "paper_reproduction": False,
    "event_count": len(data),
    "alarm_types": events,
    "device_count": len(nodes),
    "max_hop": 0,
    "max_iter": 1,
    "delta": 0.1,
    "penalty": "BIC",
    "runtime_seconds": round(elapsed, 2),
    "true_edges_off_diagonal": int(true_edges.sum()),
    "predicted_edges_off_diagonal": int(pred_edges.sum()),
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}

(output / "metrics.json").write_text(
    json.dumps(results, indent=2) + "\n"
)

print("=== RESULTS ===", flush=True)
print(json.dumps(results, indent=2), flush=True)
