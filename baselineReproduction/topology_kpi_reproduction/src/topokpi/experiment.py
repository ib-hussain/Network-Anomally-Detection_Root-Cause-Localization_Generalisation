from __future__ import annotations
import json
import random
import sys
import hashlib
import platform
import subprocess
from pathlib import Path
import numpy as np

from .data import load_npz, prepare
from .metrics import average_precision, binary_metrics, detection_delay, roc_auc, select_f1_threshold
from .rca import localize


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Install with: pip install -r requirements.txt") from exc


def load_config(path: str | Path) -> dict:
    path=Path(path)
    if not path.exists():
        candidate=Path(sys.prefix)/"share"/"topokpi"/path.name
        if candidate.exists(): path=candidate
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _sha256(path: str | Path) -> str:
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def provenance(data_path: str, config_path: str, seed: int, device: str) -> dict:
    torch=_torch()
    try:
        commit=subprocess.run(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,check=True).stdout.strip()
    except Exception: commit=None
    return {"seed":seed,"device":device,"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"torch":torch.__version__,
            "cuda":torch.version.cuda,"git_commit":commit,"data_path":str(Path(data_path).resolve()),"data_sha256":_sha256(data_path),
            "config_path":str(Path(config_path).resolve()),"config_sha256":_sha256(config_path)}


def train(data_path: str, config_path: str, output: str, seed: int = 17, device: str | None = None) -> dict:
    torch = _torch()
    from torch.utils.data import DataLoader, TensorDataset
    from .model import TopologyAwareSSL, ssl_loss
    cfg = load_config(config_path)
    _seed_everything(seed)
    raw = load_npz(data_path)
    p = prepare(raw, cfg["data"]["window_length"], cfg["data"]["stride"], tuple(cfg["data"]["split"]))
    override = cfg["data"].get("training_windows_npz")
    train_x, train_y = p.train.x, p.train.y_next
    if override:
        with np.load(override, allow_pickle=False) as z:
            train_x, train_y = z["x"].astype(np.float32), z["y_next"].astype(np.float32)
        if train_x.shape[1:] != p.train.x.shape[1:] or train_y.shape[1:] != p.train.y_next.shape[1:]:
            raise ValueError("training-window override shape does not match the target dataset")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m, tr = cfg["model"], cfg["training"]
    model = TopologyAwareSSL(raw.x.shape[-1], m["hidden_dim"], m["temporal_layers"], m["graph_layers"],
                             m["attention_heads"], m["feedforward_dim"], m["dropout"], m["topology_input_strength"]).to(device)
    a = torch.tensor(p.a_enc, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["learning_rate"])
    train_set = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(train_set, batch_size=tr["batch_size"], shuffle=True)
    validation_x = torch.from_numpy(p.validation.x).to(device)
    validation_y = torch.from_numpy(p.validation.y_next).to(device)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, tr["max_epochs"] + 1):
        model.train()
        sums = {k: 0. for k in ("loss", "reconstruction", "prediction", "topology")}
        batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mask = torch.rand_like(x) < m["mask_ratio"]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = ssl_loss(model, x, x.masked_fill(mask, 0), mask, y, a,
                                   m["lambda_rec"], m["lambda_pred"], m["lambda_topology"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"])
            optimizer.step()
            for key in sums: sums[key] += float(parts[key])
            batches += 1
        model.eval()
        with torch.no_grad():
            # Validation loss is deterministic: prediction + full-window reconstruction.
            rec, pred, latent = model(validation_x, a)
            val = float((rec - validation_x).abs().mean() + (pred - validation_y).abs().mean())
        row = {"epoch": epoch, "validation": val, **{k: v / max(batches, 1) for k, v in sums.items()}}
        history.append(row)
        if val < best - 1e-7:
            best, stale = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= tr["patience"]: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, val_node, *_ = model.scores(validation_x, validation_y, a, m["alpha"])
    val_node_np = val_node.cpu().numpy()
    val_system = np.sort(val_node_np, axis=1)[:, -min(m["top_m"], val_node_np.shape[1]):].mean(1)
    threshold, _ = select_f1_threshold(p.validation.labels, val_system)
    node_threshold = _select_node_threshold(
        val_node_np, val_system, val_system > threshold, p.validation.labels,
        p.validation.root_causes, p.validation.end_indices, raw.w_rca, cfg["rca"]
    )
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    artifact = {"model": best_state, "config": cfg, "mean": p.mean, "std": p.std, "threshold": threshold,
                "node_threshold": node_threshold, "seed": seed, "data_source": str(Path(data_path).resolve())}
    torch.save(artifact, out / "checkpoint.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance(data_path,config_path,seed,device),indent=2),encoding="utf-8")
    summary = {"best_validation_loss": best, "epochs": len(history), "threshold": threshold, "node_threshold": node_threshold, "device": device}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def event_representatives(labels: np.ndarray, roots: np.ndarray, system_scores: np.ndarray,
                          alarms: np.ndarray) -> np.ndarray:
    """Choose the highest-scoring alarm once per contiguous labeled fault event."""
    labels, roots = np.asarray(labels), np.asarray(roots)
    starts = np.flatnonzero((labels == 1) & np.r_[True, (labels[:-1] == 0) | (roots[:-1] != roots[1:])])
    selected = []
    for start in starts:
        end = start + 1
        while end < len(labels) and labels[end] == 1 and roots[end] == roots[start]:
            end += 1
        choices = np.flatnonzero(alarms[start:end]) + start
        if len(choices):
            selected.append(int(choices[np.argmax(system_scores[choices])]))
    return np.asarray(selected, dtype=int)


def _select_node_threshold(node_scores: np.ndarray, system_scores: np.ndarray, alarms: np.ndarray,
                           labels: np.ndarray, roots: np.ndarray, end_indices: np.ndarray,
                           w: np.ndarray, rca: dict) -> float:
    valid = event_representatives(labels, roots, system_scores, alarms)
    candidates = np.quantile(node_scores, np.linspace(.80, .999, 80))
    best = (-1., float(candidates[0]))
    for threshold in candidates:
        hits = []
        for idx in valid:
            raw_time = int(end_indices[idx])
            matrix = w[raw_time] if w.ndim == 3 else w
            result = localize(node_scores[idx], matrix, float(threshold), rca["candidate_cap"],
                              rca["propagation_depth"], rca["decay"], rca["gamma"])
            hits.append(bool(result.ranking and result.ranking[0] == roots[idx]))
        score = float(np.mean(hits)) if hits else 0.
        if score > best[0]: best = (score, float(threshold))
    return best[1]


def evaluate(run: str, data_path: str, device: str | None = None) -> dict:
    torch = _torch()
    from .model import TopologyAwareSSL
    artifact = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = artifact["config"]; raw = load_npz(data_path)
    p = prepare(raw, cfg["data"]["window_length"], cfg["data"]["stride"], tuple(cfg["data"]["split"]))
    m, r = cfg["model"], cfg["rca"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TopologyAwareSSL(raw.x.shape[-1], m["hidden_dim"], m["temporal_layers"], m["graph_layers"],
                             m["attention_heads"], m["feedforward_dim"], m["dropout"], m["topology_input_strength"]).to(device)
    model.load_state_dict(artifact["model"]); model.eval()
    a = torch.tensor(p.a_enc, dtype=torch.float32, device=device)
    with torch.no_grad():
        kpi, node, *_ = model.scores(torch.from_numpy(p.test.x).to(device), torch.from_numpy(p.test.y_next).to(device), a, m["alpha"])
    kpi, node = kpi.cpu().numpy(), node.cpu().numpy()
    system = np.sort(node, axis=1)[:, -min(m["top_m"], node.shape[1]):].mean(1)
    binary = binary_metrics(p.test.labels, system, artifact["threshold"])
    binary.update({"auroc": roc_auc(p.test.labels, system), "auprc": average_precision(p.test.labels, system),
                   "delay": detection_delay(p.test.labels, system > artifact["threshold"])})
    rankings, roots = [], []
    event_indices = event_representatives(p.test.labels, p.test.root_causes, system, system > artifact["threshold"])
    for q in event_indices:
        raw_idx = p.test.end_indices[q]
        matrix = raw.w_rca[raw_idx] if raw.w_rca.ndim == 3 else raw.w_rca
        result = localize(node[q], matrix, artifact["node_threshold"], r["candidate_cap"], r["propagation_depth"], r["decay"], r["gamma"], kpi[q])
        rankings.append(result.ranking); roots.append(int(p.test.root_causes[q]))
    from .metrics import rca_metrics
    metrics = {"detection": binary, "rca": rca_metrics(rankings, roots) if roots else {}, "events": len(roots)}
    Path(run, "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    padded_rankings=np.full((len(rankings),raw.x.shape[1]),-1,np.int32)
    for i,ranking in enumerate(rankings): padded_rankings[i,:len(ranking)]=ranking
    np.savez_compressed(Path(run, "test_outputs.npz"), system_scores=system, node_scores=node, kpi_scores=kpi,
                        labels=p.test.labels, root_causes=p.test.root_causes, end_indices=p.test.end_indices,
                        fault_types=p.test.fault_types.astype(str), event_indices=event_indices, event_rankings=padded_rankings,
                        event_roots=np.asarray(roots,np.int32), event_fault_types=p.test.fault_types[event_indices].astype(str),
                        alarm_threshold=np.asarray(artifact["threshold"]), node_threshold=np.asarray(artifact["node_threshold"]))
    return metrics
