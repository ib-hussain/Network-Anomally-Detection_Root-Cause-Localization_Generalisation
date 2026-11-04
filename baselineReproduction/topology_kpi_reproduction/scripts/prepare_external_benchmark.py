#!/usr/bin/env python3
"""Convert the public DejaVu-style benchmark to the topokpi npz schema.

Manuscript Section 5.3 evaluates external validity on Dataset D of the public
DejaVu benchmark (Li et al., FSE'22, https://github.com/NetManAIOps/DejaVu).
That benchmark's own README confirms each dataset folder contains:
  - `graph.yml` or `graphs/*.yml`: the failure dependency graph (FDG)
  - `metrics.csv`: per-node metric time series
  - `faults.csv`: failure events with ground-truth root causes

To accommodate differences among archived releases:

  1. Run with --inspect first. It prints the discovered structure
     (yaml top-level keys, CSV columns, dtypes, a few sample rows) and writes
     nothing. Compare that output against what this docstring assumes.
  2. If column names differ, pass the --time-col / --node-col / --metric-col
     / --value-col / --start-col / --end-col / --root-col / --class-col
     overrides, or edit the small `_pick` heuristics below.
  3. Then drop --inspect to write the npz file consumed by
     `topokpi train` / `topokpi evaluate` unmodified.

Requires pandas and pyyaml, which are NOT in requirements.txt because they are
only needed for this optional external-benchmark adapter:
    pip install pandas pyyaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from topokpi.data import RawDataset, save_npz


def _lazy_imports():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required: pip install pandas") from exc
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("pyyaml is required: pip install pyyaml") from exc
    return pd, yaml


def _pick(candidates: list[str], columns: list[str], override: str | None, purpose: str) -> str:
    if override:
        if override not in columns:
            raise SystemExit(f"--{purpose}-col={override!r} not found in columns {columns}")
        return override
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    raise SystemExit(
        f"Could not auto-detect the {purpose} column among {columns}. "
        f"Re-run with --inspect to see the raw data, then pass --{purpose}-col explicitly."
    )


def load_graph(data_dir: Path, yaml_module) -> tuple[list[str], list[tuple[str, str]], bool]:
    """Return (node_names, edges, directed). Tries graph.yml then graphs/*.yml."""
    candidates = [data_dir / "graph.yml"]
    candidates += sorted((data_dir / "graphs").glob("*.yml")) if (data_dir / "graphs").is_dir() else []
    for path in candidates:
        if not path.exists():
            continue
        payload = yaml_module.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "nodes" in payload and "edges" in payload:
            nodes = [str(n) for n in payload["nodes"]]
            edges = [(str(u), str(v)) for u, v in payload["edges"]]
            directed = bool(payload.get("directed", False))
            return nodes, edges, directed
        if isinstance(payload, dict):
            # Adjacency-list style: {node: [neighbors...]}
            nodes = [str(n) for n in payload.keys()]
            edges = [(str(u), str(v)) for u, vs in payload.items() for v in (vs or [])]
            return nodes, edges, True
        raise SystemExit(f"Unrecognized graph YAML shape in {path}: expected a dict with 'nodes'/'edges'.")
    raise SystemExit(f"No graph.yml or graphs/*.yml found under {data_dir}")


def inspect(data_dir: Path) -> None:
    pd, yaml_module = _lazy_imports()
    print(f"--- {data_dir} ---")
    for name in ("graph.yml", "metrics.csv", "faults.csv"):
        path = data_dir / name
        print(f"{name}: {'found' if path.exists() else 'MISSING'}")
    graphs_dir = data_dir / "graphs"
    if graphs_dir.is_dir():
        print(f"graphs/: {sorted(p.name for p in graphs_dir.glob('*.yml'))}")
    for name in ("metrics.csv", "faults.csv"):
        path = data_dir / name
        if path.exists():
            frame = pd.read_csv(path, nrows=5)
            print(f"\n{name} columns: {list(frame.columns)}")
            print(frame.head())
    graph_path = data_dir / "graph.yml"
    if graph_path.exists():
        payload = yaml_module.safe_load(graph_path.read_text(encoding="utf-8"))
        keys = list(payload.keys()) if isinstance(payload, dict) else type(payload)
        print(f"\ngraph.yml top-level keys: {keys}")


def convert(args) -> None:
    pd, yaml_module = _lazy_imports()
    data_dir = Path(args.data_dir)
    nodes, edges, directed = load_graph(data_dir, yaml_module)
    node_index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)

    a_enc = np.zeros((n, n), np.float32)
    w_rca = np.zeros((n, n), np.float32)
    for u, v in edges:
        if u not in node_index or v not in node_index:
            continue
        ui, vi = node_index[u], node_index[v]
        a_enc[ui, vi] = a_enc[vi, ui] = 1.0
        w_rca[ui, vi] = 1.0
        if not directed:
            w_rca[vi, ui] = 1.0

    metrics = pd.read_csv(data_dir / "metrics.csv")
    columns = list(metrics.columns)
    time_col = _pick(["timestamp", "time", "ts"], columns, args.time_col, "time")
    node_col = _pick(["node", "failure_unit", "instance", "service", "entity"], columns, args.node_col, "node")
    metric_col = _pick(["metric", "name", "kpi", "metric_name"], columns, args.metric_col, "metric")
    value_col = _pick(["value", "val"], columns, args.value_col, "value")

    metrics = metrics[metrics[node_col].astype(str).isin(node_index)]
    kpi_names = sorted(metrics[metric_col].astype(str).unique())[: args.max_kpis]
    metrics = metrics[metrics[metric_col].astype(str).isin(kpi_names)]
    kpi_index = {name: i for i, name in enumerate(kpi_names)}
    timestamps = np.sort(metrics[time_col].unique())
    time_index = {t: i for i, t in enumerate(timestamps)}

    x = np.zeros((len(timestamps), n, len(kpi_names)), np.float32)
    filled = np.zeros_like(x, dtype=bool)
    for row in metrics.itertuples(index=False):
        ti = time_index[getattr(row, time_col)]
        ni = node_index[str(getattr(row, node_col))]
        ki = kpi_index[str(getattr(row, metric_col))]
        x[ti, ni, ki] = getattr(row, value_col)
        filled[ti, ni, ki] = True
    # Forward-fill missing (node, kpi) entries along time, then zero-fill the rest.
    for ni in range(n):
        for ki in range(len(kpi_names)):
            column = x[:, ni, ki]
            mask = filled[:, ni, ki]
            last = 0.0
            for t in range(len(column)):
                if mask[t]:
                    last = column[t]
                else:
                    column[t] = last

    faults = pd.read_csv(data_dir / "faults.csv")
    fcolumns = list(faults.columns)
    start_col = _pick(["start_time", "start", "timestamp"], fcolumns, args.start_col, "start")
    end_col = _pick(["end_time", "end", "finish_time"], fcolumns, args.end_col, "end")
    root_col = _pick(["root_cause", "root_cause_node", "failure_unit", "instance"], fcolumns, args.root_col, "root")
    class_col = args.class_col or next((c for c in fcolumns if "class" in c.lower() or "type" in c.lower()), None)

    labels = np.zeros(len(timestamps), np.int8)
    root_causes = np.full(len(timestamps), -1, np.int32)
    fault_types = np.full(len(timestamps), "normal", dtype="<U64")
    for row in faults.itertuples(index=False):
        start, end = getattr(row, start_col), getattr(row, end_col)
        root_name = str(getattr(row, root_col)).split(";")[0].strip()
        if root_name not in node_index:
            continue
        lo = np.searchsorted(timestamps, start, side="left")
        hi = np.searchsorted(timestamps, end, side="right")
        labels[lo:hi] = 1
        root_causes[lo:hi] = node_index[root_name]
        fault_types[lo:hi] = str(getattr(row, class_col)) if class_col else "unknown"

    dataset = RawDataset(x, labels, root_causes, fault_types, a_enc, w_rca, timestamps.astype(np.int64))
    save_npz(args.output, dataset)
    print(f"Wrote {args.output}: x={x.shape}, nodes={n}, kpis={len(kpi_names)}, "
          f"labeled_steps={int(labels.sum())}/{len(labels)}. "
          "Verify the converted columns and spot-check several events before evaluation.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Extracted DejaVu-style dataset folder (e.g. data/D)")
    parser.add_argument("--output", help="Destination npz path (required unless --inspect)")
    parser.add_argument("--inspect", action="store_true", help="Print discovered structure and exit; write nothing")
    parser.add_argument("--max-kpis", type=int, default=32, help="Cap on distinct metric names kept as KPI dimensions")
    for flag in ("time-col", "node-col", "metric-col", "value-col", "start-col", "end-col", "root-col", "class-col"):
        parser.add_argument(f"--{flag}", dest=flag.replace("-", "_"))
    args = parser.parse_args()
    if args.inspect:
        inspect(Path(args.data_dir))
        return
    if not args.output:
        raise SystemExit("--output is required unless --inspect is set")
    convert(args)


if __name__ == "__main__":
    main()
