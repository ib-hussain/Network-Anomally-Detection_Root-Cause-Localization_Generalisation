"""Calculate Zhou et al. (2026) Eq. (5) ACG weights for a supplied causal graph.

This weights an ALREADY learned directed event-type graph. It does NOT learn
that graph, fit TTHP, or provide physical-topology/root-cause labels.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def presence_in_time_slices(events, n_types, n_bins, train_fraction=0.7):
    """Use only the first fraction of the *timestamp range* to build slices.

    Slice intervals are [start + j*width, start + (j+1)*width), with the
    overall training interval [min_time, cutoff). This is an explicit
    experimental convention; the paper does not fix the bin width.
    """
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")
    needed = {"event", "timestamp", "node"}
    if not needed.issubset(events.columns):
        raise ValueError(f"Expected columns {sorted(needed)}")
    if events.empty:
        raise ValueError("No event rows provided")
    ts = pd.to_numeric(events["timestamp"], errors="coerce").to_numpy(dtype=float)
    event_num = pd.to_numeric(events["event"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(ts).all() or not np.isfinite(event_num).all():
        raise ValueError("Timestamps and event types must be finite numeric values")
    if not ((event_num >= 0) & (event_num < n_types) & (event_num == np.floor(event_num))).all():
        raise ValueError("Event type indices must be integers from 0 to n_types-1")
    start, finish = float(ts.min()), float(ts.max())
    if not finish > start:
        raise ValueError("Timestamp range must be nonzero")
    cutoff = start + train_fraction * (finish - start)
    train_mask = ts < cutoff
    if not train_mask.any():
        raise ValueError("No training events fall before the timestamp cutoff")
    # This step does not use test observations to construct the weights.
    width = (cutoff - start) / n_bins
    indices = np.floor((ts[train_mask] - start) / width).astype(int)
    # Floating-point rounding at the boundary is explicitly handled.
    indices = np.clip(indices, 0, n_bins - 1)
    presence = np.zeros((n_bins, n_types), dtype=bool)
    presence[indices, event_num[train_mask].astype(int)] = True
    metadata = {
        "train_time_start": start,
        "train_time_cutoff_exclusive": cutoff,
        "observation_time_max": finish,
        "time_slice_width_raw_timestamp_units": width,
        "train_event_count": int(train_mask.sum()),
        "nontrain_event_count": int((~train_mask).sum()),
        "time_slice_count": n_bins,
        "train_time_fraction": train_fraction,
    }
    return presence, metadata


def conditional_acg_weights(presence, directed_graph):
    """Eq. (5): weight[i,j] = # slices with i and j / # slices with i.

    Only the edges in directed_graph are assigned weights. No new edges are
    introduced. Zero-support source types receive weight zero, not NaN.
    """
    p = np.asarray(presence)
    a = np.asarray(directed_graph)
    if p.ndim != 2 or p.shape[0] < 1:
        raise ValueError("presence must have shape [time_slices,event_types]")
    if not np.isfinite(p).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("presence must contain only 0 or 1")
    n_types = p.shape[1]
    if a.shape != (n_types, n_types) or not np.isfinite(a).all():
        raise ValueError("directed graph must be a finite square [event_types,event_types] array")
    if not np.isin(a, [0, 1]).all():
        raise ValueError("directed causal graph must be binary (0/1), not weighted")
    if np.any(np.diag(a) != 0):
        raise ValueError("directed causal graph must have no self-loops")
    binary = p.astype(np.int64)
    support = binary.sum(axis=0)
    joint = binary.T @ binary
    weights = np.divide(
        joint, support[:, None],
        out=np.zeros((n_types, n_types), dtype=float),
        where=support[:, None] > 0,
    )
    weights *= a.astype(bool)
    np.fill_diagonal(weights, 0)
    if not np.all((weights >= 0) & (weights <= 1)):
        raise AssertionError("ACG conditional weights outside [0,1]")
    return weights, support


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for piece in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--events", default="data/processed/wireless_thp_events.csv")
    cli.add_argument("--graph", required=True, help="LEARNED directed event-type graph (.npy); never DAG.npy")
    cli.add_argument("--bins", type=int, default=384, help="Exploratory training-time slices; paper width unverified")
    cli.add_argument("--train-fraction", type=float, default=0.7, help="Fraction of timestamp RANGE, not fraction of rows")
    cli.add_argument("--output", default="outputs/acg_eq5_exploratory")
    args = cli.parse_args()
    graph_path = Path(args.graph)
    if graph_path.name.lower() == "dag.npy":
        cli.error("DAG.npy is ground truth. It must NOT be supplied as the learned graph.")
    if not graph_path.is_file():
        cli.error(f"Learned graph file does not exist: {graph_path}")
    event_path = Path(args.events)
    if not event_path.is_file():
        cli.error(f"Event file does not exist: {event_path}")
    graph = np.load(graph_path, allow_pickle=False)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        cli.error("Learned graph must be square")
    events = pd.read_csv(event_path)
    presence, summary = presence_in_time_slices(events, graph.shape[0], args.bins, args.train_fraction)
    weights, support = conditional_acg_weights(presence, graph)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        cli.error(f"Output already exists: {output}; use a new --output directory")
    np.save(output / "acg_weights.npy", weights)
    pd.DataFrame(
        [(int(i), int(j), float(weights[i, j]), int(support[i]))
         for i, j in np.argwhere(graph != 0)],
        columns=["source_event", "target_event", "conditional_weight", "source_slice_support"],
    ).to_csv(output / "edges.csv", index=False)
    summary.update({
        "label": "Eq5 exploratory weighting; NOT TTHP or ACG-LGraphBM reproduction",
        "learned_graph_file": str(graph_path),
        "learned_graph_sha256": sha256(graph_path),
        "events_file": str(event_path),
        "events_sha256": sha256(event_path),
        "graph_edge_count": int((graph != 0).sum()),
        "weighted_nonzero_edge_count": int((weights > 0).sum()),
        "zero_support_event_types": np.flatnonzero(support == 0).tolist(),
        "source_graph_may_have_seen_full_dataset": True,
        "paper_time_slice_width_unverified": True,
        "paper_reproduction": False,
    })
    (output / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"ACG weighted edges saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
