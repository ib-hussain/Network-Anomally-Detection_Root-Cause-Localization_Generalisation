"""Paper-grounded ACG threshold sampling; NO topology/data fabrication.

Modes:
  preview: Show event-type ACG edges surviving a threshold. It is *not*
           a fault-case alarm-correlation subgraph (ACSG).
  case:    Build one ACSG from an explicitly supplied directed node topology
           and a single fault case's per-node alarm observations.

For a physical topology edge node_i -> node_j with representative
alarm types v_i and v_j, Zhou et al. section 4.3 uses ACG[v_j, v_i].
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_acg(path):
    """Load precomputed directed type-type weights; NEVER the ground-truth DAG."""
    path = Path(path)
    if path.name.lower() == "dag.npy":
        raise ValueError("DAG.npy is ground truth, not a learned ACG")
    w = np.load(path, allow_pickle=False)
    if (w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2 or
            not np.isfinite(w).all() or (w < 0).any() or (w > 1).any() or
            np.any(np.diag(w) != 0)):
        raise ValueError("ACG must be finite, square, off-diagonal, and within [0,1]")
    return w


def validate_threshold(threshold):
    """Use strictly positive thresholds; paper's zero threshold discussion is ambiguous."""
    if not np.isfinite(threshold) or not (0 < threshold <= 1):
        raise ValueError("threshold must be greater than 0 and at most 1")


def preview_event_edges(acg, threshold):
    """For diagnostics ONLY: list directed event-type edges with weight >= threshold."""
    validate_threshold(threshold)
    w = np.asarray(acg)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or not np.isfinite(w).all():
        raise ValueError("Invalid ACG")
    return [(int(i), int(j), float(w[i, j]))
            for i, j in np.argwhere((w >= threshold) & ~np.eye(len(w), dtype=bool))]


def dominant_alarm_types(alarms, num_nodes, num_types):
    """Representative alarm per physical node: most frequent; min ID breaks ties.

    Node without an alarm is represented by -1. A single fault case only.
    """
    if not {"node_id", "event_type"}.issubset(alarms.columns):
        raise ValueError("Case CSV needs columns node_id,event_type")
    result = np.full(num_nodes, -1, dtype=int)
    if alarms.empty:
        return result
    for col, upper in (("node_id", num_nodes), ("event_type", num_types)):
        vals = pd.to_numeric(alarms[col], errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(vals).all() or (vals != np.floor(vals)).any()
                or (vals < 0).any() or (vals >= upper).any()):
            raise ValueError("Invalid case alarm " + col + "; use zero-based integer IDs")
    counts = alarms.groupby(["node_id", "event_type"]).size().reset_index(name="count")
    counts = counts.sort_values(["node_id", "count", "event_type"],
                                ascending=[True, False, True])
    winners = counts.drop_duplicates("node_id")
    for _, record in winners.iterrows():
        result[int(record["node_id"])] = int(record["event_type"])
    return result


def sample_case_edges(topology, node_types, acg, threshold):
    """Section 4.3: preserve strong edges within *one case's* alarm-node subgraph.

    Directed topology[i,j] = 1 is node_i -> node_j. Per the paper, lookup
    ACG[alarm_type_at_j, alarm_type_at_i] (reverse direction).
    Returns (node_i, node_j, event_j, event_i, weight).
    """
    validate_threshold(threshold)
    w = np.asarray(acg, dtype=float)
    t = np.asarray(topology)
    node_types = np.asarray(node_types, dtype=int)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or len(node_types) != t.shape[0]:
        raise ValueError("Topology must be square, with one alarm type per node")
    if not np.isfinite(t).all() or not np.isin(t, (0, 1)).all() or (np.diag(t) != 0).any():
        raise ValueError("Topology must be binary, directed, and without self-loops")
    if w.ndim != 2 or w.shape[0] != w.shape[1] or not np.isfinite(w).all():
        raise ValueError("Invalid ACG")
    if ((node_types < -1) | (node_types >= w.shape[0])).any():
        raise ValueError("Node alarm types out of ACG range")
    retained = []
    for i, j in np.argwhere(t == 1):
        vi, vj = int(node_types[i]), int(node_types[j])
        if vi < 0 or vj < 0:
            continue  # paper restricts initial topology subgraph to alarmed nodes
        weight = float(w[vj, vi])  # intentional reverse-alarm direction
        if weight >= threshold:
            retained.append((int(i), int(j), vj, vi, weight))
    return retained


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acg", required=True, help="ACG weighted adjacency .npy; NOT DAG.npy")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="0.7 is paper's Suning threshold; not optimized for Wireless")
    p.add_argument("--mode", choices=("preview", "case"), default="preview")
    p.add_argument("--topology", help="Case-mode only: physical directed NxN headerless 0/1 CSV")
    p.add_argument("--alarms", help="Case-mode only: one case CSV with node_id,event_type rows")
    p.add_argument("--output", required=True, help="New output directory (must not already exist)")
    args = p.parse_args()
    acg_path = Path(args.acg)
    if not acg_path.is_file():
        p.error("ACG file not found")
    try:
        w = load_acg(acg_path)
        validate_threshold(args.threshold)
        if args.mode == "case":
            if not args.topology or not args.alarms:
                p.error("--mode case requires both --topology and --alarms")
            topo_path, alarm_path = Path(args.topology), Path(args.alarms)
            topology = np.loadtxt(topo_path, delimiter=",", ndmin=2)
            alarms = pd.read_csv(alarm_path)
            kinds = dominant_alarm_types(alarms, topology.shape[0], len(w))
            edges = sample_case_edges(topology, kinds, w, args.threshold)
            columns = ["node_i", "node_j", "alarm_j", "alarm_i", "acg_weight"]
            metadata = {"mode": "case", "topology_file": str(topo_path),
                        "topology_sha256": sha256(topo_path),
                        "case_alarm_file": str(alarm_path),
                        "case_alarm_sha256": sha256(alarm_path),
                        "nodes_with_alarm": int((kinds >= 0).sum()),
                        "topology_directed_edges": int((topology == 1).sum())}
        else:
            if args.topology or args.alarms:
                p.error("--topology/--alarms require --mode case")
            edges = preview_event_edges(w, args.threshold)
            columns = ["source_event", "target_event", "acg_weight"]
            metadata = {"mode": "preview", "note": "Event-type preview ONLY; not a physical topology or ACSG"}
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    output = Path(args.output)
    if output.exists():
        p.error(f"Output already exists: {output}. Choose a new directory")
    output.mkdir(parents=True)
    df = pd.DataFrame(edges, columns=columns)
    df.to_csv(output / "retained_edges.csv", index=False)
    metadata.update({"paper": "Zhou et al. (2026) section 4.3",
                     "threshold": args.threshold,
                     "acg_file": str(acg_path), "acg_sha256": sha256(acg_path),
                     "retained_edge_count": len(edges),
                     "paper_reproduction": False,
                     "paper_threshold_context": "0.7 selected on Suning; not validated for Wireless"})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(df.to_string(index=False) if not df.empty else "No retained edges")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
