import json
from pathlib import Path

import numpy as np
import pandas as pd

data = pd.read_csv(
    "data/processed/wireless_thp_events.csv"
)

pred = np.load("outputs/thp_smoke_graph.npy")

true_full = np.load(
    "data/raw/wireless/18V_55N_Wireless/DAG.npy",
    allow_pickle=False
)

# Reconstruct the exact selection used for the smoke test.
selected_events = data["event"].value_counts().head(4).index
subset = data[data["event"].isin(selected_events)].copy()

selected_nodes = subset["node"].value_counts().head(5).index
subset = subset[subset["node"].isin(selected_nodes)]
subset = subset.sort_values("timestamp").head(1000)

event_ids = sorted(subset["event"].unique().tolist())

assert pred.shape == (len(event_ids), len(event_ids))
assert max(event_ids) < true_full.shape[0]

true = true_full[np.ix_(event_ids, event_ids)] != 0
pred = pred != 0

# Exclude diagonal self-connections from directed-edge scoring.
np.fill_diagonal(true, False)
np.fill_diagonal(pred, False)

tp = int(np.logical_and(pred, true).sum())
fp = int(np.logical_and(pred, ~true).sum())
fn = int(np.logical_and(~pred, true).sum())

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

results = {
    "experiment": "THP smoke test, not TTHP",
    "alarm_types": event_ids,
    "events_used": len(subset),
    "max_hop": 0,
    "max_iter": 1,
    "true_edges": int(true.sum()),
    "predicted_edges": int(pred.sum()),
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}

print("=== CAUSAL GRAPH EVALUATION ===")
print(json.dumps(results, indent=2))

Path("outputs").mkdir(exist_ok=True)
Path("outputs/thp_smoke_metrics.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
