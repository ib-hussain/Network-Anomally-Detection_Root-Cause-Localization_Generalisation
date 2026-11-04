#!/usr/bin/env python3
"""Run the manuscript's five-seed protocol and aggregate recomputed metrics."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np

from topokpi.experiment import evaluate, load_config, train


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    # t(0.975,4)=2.776 for the manuscript's fixed five seeds.
    critical = 2.776 if len(array) == 5 else 1.96
    half = critical * sd / math.sqrt(len(array))
    return {"mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seeds = cfg.get("training", {}).get("seeds", [17, 29, 43, 71, 101])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        run = output / f"seed_{seed}"
        train(args.data, args.config, str(run), seed, args.device)
        metrics = evaluate(str(run), args.data, args.device)
        runs.append({"seed": seed, **metrics})
    keys = (("detection", "f1"), ("detection", "auroc"), ("detection", "auprc"),
            ("rca", "a@1"), ("rca", "a@3"), ("rca", "a@5"), ("rca", "mrr"), ("rca", "mean_rank"))
    summary = {f"{group}.{metric}": summarize([r[group][metric] for r in runs]) for group, metric in keys}
    payload = {"runs": runs, "summary": summary}
    (output / "five_seed_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
