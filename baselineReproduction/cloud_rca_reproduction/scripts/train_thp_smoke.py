from pathlib import Path
import numpy as np
import pandas as pd
from castle.algorithms import TTPM

data = pd.read_csv(
    "data/processed/wireless_thp_events.csv"
)

# Keep only a few common event types and devices
# so the first integration test is small.
selected_events = data["event"].value_counts().head(4).index
subset = data[data["event"].isin(selected_events)].copy()

selected_nodes = subset["node"].value_counts().head(5).index
subset = subset[subset["node"].isin(selected_nodes)].copy()

subset = subset.sort_values("timestamp").head(1000).copy()

node_ids = sorted(subset["node"].unique())
node_map = {node: i for i, node in enumerate(node_ids)}
subset["node"] = subset["node"].map(node_map)

assert len(subset) > 10
assert subset["event"].nunique() >= 2

# Identity is used ONLY for this no-neighbor smoke test.
# It is NOT the physical topology of the wireless network.
topology = np.eye(len(node_ids), dtype=int)

print("Training rows:", len(subset), flush=True)
print("Event types:", sorted(subset["event"].unique()), flush=True)
print("Devices:", len(node_ids), flush=True)

model = TTPM(
    topology_matrix=topology,
    max_hop=0,
    max_iter=1,
)

model.learn(subset[["event", "timestamp", "node"]].copy())

graph = np.asarray(model.causal_matrix)
Path("outputs").mkdir(exist_ok=True)
np.save("outputs/thp_smoke_graph.npy", graph)

print("Learned graph shape:", graph.shape)
print("Saved outputs/thp_smoke_graph.npy")
print("SMOKE TEST ONLY — not the paper's TTHP model.")
