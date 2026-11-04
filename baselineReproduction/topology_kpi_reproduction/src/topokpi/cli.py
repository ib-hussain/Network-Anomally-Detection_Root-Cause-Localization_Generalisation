from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

from .data import generate_demo, load_npz, save_npz


def _print(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def command_generate(args) -> None:
    data = generate_demo(args.nodes, args.steps, args.seed, args.event_start)
    save_npz(args.output, data)
    _print({"written": str(Path(args.output).resolve()), "shape": list(data.x.shape), "warning": "synthetic pipeline fixture; not paper evidence"})


def command_train(args) -> None:
    from .experiment import train
    _print(train(args.data, args.config, args.output, args.seed, args.device))


def command_evaluate(args) -> None:
    from .experiment import evaluate
    _print(evaluate(args.run, args.data, args.device))


def command_baseline(args) -> None:
    from .baselines import method_from_name
    from .metrics import average_precision, binary_metrics, roc_auc, select_f1_threshold
    raw = load_npz(args.data)
    t, n, d = raw.x.shape
    train_stop, val_stop = int(.60 * t), int(.75 * t)
    mean, std = raw.x[:train_stop].mean((0, 1), keepdims=True), raw.x[:train_stop].std((0, 1), keepdims=True)
    x = (raw.x - mean) / np.maximum(std, 1e-6)
    model = method_from_name(args.name).fit(x[:train_stop])
    val_system, _ = model.score(x[train_stop-1:val_stop], n, d)
    threshold, val = select_f1_threshold(raw.labels[train_stop:val_stop], val_system)
    test_system, node = model.score(x[val_stop-1:], n, d)
    metrics = binary_metrics(raw.labels[val_stop:], test_system, threshold)
    metrics.update({"auroc": roc_auc(raw.labels[val_stop:], test_system), "auprc": average_precision(raw.labels[val_stop:], test_system)})
    from .experiment import event_representatives
    event_indices = event_representatives(raw.labels[val_stop:], raw.root_causes[val_stop:], test_system, test_system > threshold)
    rankings, roots = [], []
    for q in event_indices:
        rankings.append(np.argsort(-node[q], kind="stable").tolist())
        roots.append(int(raw.root_causes[val_stop + q]))
    from .metrics import rca_metrics
    _print({"method": args.name, "threshold": threshold, "validation": val, "test_detection": metrics,
            "test_residual_rca": rca_metrics(rankings, roots) if roots else {}})


def command_deep_baseline(args) -> None:
    from .baseline_experiment import run_deep_baseline
    _print(run_deep_baseline(args.name,args.data,args.config,args.output,args.seed,args.device,args.epochs))


def command_reported(args) -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "reported" / "manuscript_results.json"
    if not path.exists(): path=Path(sys.prefix)/"share"/"topokpi"/"manuscript_results.json"
    if not path.exists(): raise SystemExit("manuscript_results.json was not installed")
    with path.open(encoding="utf-8") as handle:
        _print(json.load(handle))


def command_validate(args) -> None:
    data = load_npz(args.data)
    _print({"valid": True, "shape": list(data.x.shape), "anomaly_timepoint_ratio": float(data.labels.mean()),
            "labeled_root_points": int(np.sum(data.root_causes >= 0)), "dynamic_rca_topology": data.w_rca.ndim == 3})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="topokpi")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("generate-demo"); q.add_argument("--output", required=True); q.add_argument("--nodes", type=int, default=12); q.add_argument("--steps", type=int, default=1200); q.add_argument("--seed", type=int, default=17); q.add_argument("--event-start",type=float,default=.62); q.set_defaults(func=command_generate)
    q = sub.add_parser("validate-data"); q.add_argument("--data", required=True); q.set_defaults(func=command_validate)
    q = sub.add_parser("train"); q.add_argument("--data", required=True); q.add_argument("--config", default="configs/paper.json"); q.add_argument("--output", required=True); q.add_argument("--seed", type=int, default=17); q.add_argument("--device"); q.set_defaults(func=command_train)
    q = sub.add_parser("evaluate"); q.add_argument("--run", required=True); q.add_argument("--data", required=True); q.add_argument("--device"); q.set_defaults(func=command_evaluate)
    q = sub.add_parser("baseline"); q.add_argument("--name", choices=["OLS", "VAR(1)"], required=True); q.add_argument("--data", required=True); q.set_defaults(func=command_baseline)
    deep=["OmniAnomaly","MSCRED","USAD","THOC","MTAD-GAT","GDN","InterFusion","Anomaly Transformer","TranAD","DCdetector"]
    q = sub.add_parser("deep-baseline"); q.add_argument("--name",choices=deep,required=True); q.add_argument("--data",required=True); q.add_argument("--config",default="configs/paper.json"); q.add_argument("--output",required=True); q.add_argument("--seed",type=int,default=17); q.add_argument("--device"); q.add_argument("--epochs",type=int); q.set_defaults(func=command_deep_baseline)
    q = sub.add_parser("reported-results"); q.set_defaults(func=command_reported)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
