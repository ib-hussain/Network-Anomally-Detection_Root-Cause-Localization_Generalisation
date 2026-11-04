#!/usr/bin/env python3
"""Run and aggregate the revision experiments behind manuscript Tables 13--16.

The manifest supplies one target dataset and one disjoint abnormal-window pool
for each topology scale. Results are macro-averaged across scales within each
aligned seed, then summarized across the five seeds with sample SD and a
Student-t 95% confidence interval.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


METRICS = {
    "f1": ("detection", "f1"),
    "auroc": ("detection", "auroc"),
    "a@1": ("rca", "a@1"),
    "a@3": ("rca", "a@3"),
}


def resolved(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    critical = 2.776 if len(array) == 5 else 1.96
    half = critical * sd / math.sqrt(len(array))
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def load_result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing experiment output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric(payload: dict, name: str) -> float:
    group, key = METRICS[name]
    return float(payload[group][key])


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--device")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in manifest.get("seeds", [17, 29, 43, 71, 101])]
    scales = manifest["scales"]
    if sorted(int(item["nodes"]) for item in scales) != [12, 24, 48]:
        raise SystemExit("manifest must contain exactly the 12-, 24-, and 48-node scales")
    if len(seeds) != 5:
        raise SystemExit("the revision protocol requires exactly five aligned seeds")

    repository = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = resolved(args.config, Path.cwd())

    records = []
    for scale in scales:
        nodes = int(scale["nodes"])
        data = resolved(scale["data"], manifest_path.parent)
        pool = resolved(scale["abnormal_pool"], manifest_path.parent)
        for seed in seeds:
            root = output / f"nodes_{nodes}" / f"seed_{seed}"
            sweep_dir = root / "sweeps"
            contamination_dir = root / "contamination"
            if not args.aggregate_only:
                common = ["--config", str(config), "--seed", str(seed)]
                if args.device:
                    common += ["--device", args.device]
                subprocess.run(
                    [sys.executable, str(repository / "scripts/run_ablation_sensitivity.py"),
                     "--data", str(data), "--output", str(sweep_dir), *common],
                    cwd=repository, check=True,
                )
                subprocess.run(
                    [sys.executable, str(repository / "scripts/run_contamination.py"),
                     "--data", str(data), "--abnormal-pool", str(pool),
                     "--output", str(contamination_dir), *common],
                    cwd=repository, check=True,
                )
            for family, result_path in (
                ("sweep", sweep_dir / "results.json"),
                ("contamination", contamination_dir / "results.json"),
            ):
                for condition, payload in load_result(result_path).items():
                    records.append({
                        "family": family,
                        "condition": condition,
                        "nodes": nodes,
                        "seed": seed,
                        **{name: metric(payload, name) for name in METRICS},
                    })

    # First macro-average the three topology scales for each seed and condition.
    per_seed = []
    keys = sorted({(row["family"], row["condition"], row["seed"]) for row in records})
    for family, condition, seed in keys:
        selected = [row for row in records if (row["family"], row["condition"], row["seed"]) == (family, condition, seed)]
        if len(selected) != 3:
            raise RuntimeError(f"expected three scales for {family}/{condition}/seed {seed}")
        per_seed.append({"family": family, "condition": condition, "seed": seed,
                         **{name: float(np.mean([row[name] for row in selected])) for name in METRICS}})

    summaries = []
    for family, condition in sorted({(row["family"], row["condition"]) for row in per_seed}):
        selected = [row for row in per_seed if (row["family"], row["condition"]) == (family, condition)]
        row = {"family": family, "condition": condition}
        for name in METRICS:
            for suffix, value in summarize([item[name] for item in selected]).items():
                row[f"{name}_{suffix}"] = value
        summaries.append(row)

    write_table(output / "revision_event_records.csv", records)
    write_table(output / "revision_per_seed_macro.csv", per_seed)
    write_table(output / "revision_summary.csv", summaries)
    (output / "revision_summary.json").write_text(
        json.dumps({"protocol": {"seeds": seeds, "scales": [12, 24, 48], "single_root": True,
                                  "scale_aggregation": "macro within seed"}, "rows": summaries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(records), "summary_rows": len(summaries), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
