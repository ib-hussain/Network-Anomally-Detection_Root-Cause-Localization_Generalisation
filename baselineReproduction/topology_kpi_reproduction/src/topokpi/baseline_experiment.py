from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load_npz, prepare
from .deep_baselines import build_deep_baseline
from .experiment import load_config
from .metrics import average_precision, binary_metrics, detection_delay, roc_auc, select_f1_threshold


def _score_batches(model, windows, targets, batch: int, device: str) -> np.ndarray:
    loader=DataLoader(TensorDataset(torch.from_numpy(windows),torch.from_numpy(targets)),batch_size=batch)
    values=[]; model.eval()
    with torch.no_grad():
        for x,y in loader: values.append(model.score(x.to(device),y.to(device)).detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0)


def run_deep_baseline(name: str, data_path: str, config_path: str, output: str,
                      seed: int = 17, device: str | None = None, epochs: int | None = None) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    cfg=load_config(config_path); raw=load_npz(data_path); dc=cfg["data"]
    p=prepare(raw,dc["window_length"],dc["stride"],tuple(dc["split"])); features=raw.x.shape[1]*raw.x.shape[2]
    model=build_deep_baseline(name,features,dc["window_length"]); device=device or ("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    train_cfg=cfg["training"]; max_epochs=epochs or train_cfg["max_epochs"]
    # Signature matrices are quadratic in features; cap only the mini-batch, not the data or objective.
    batch=min(train_cfg["batch_size"],4) if name=="MSCRED" else train_cfg["batch_size"]
    loader=DataLoader(TensorDataset(torch.from_numpy(p.train.x),torch.from_numpy(p.train.y_next)),batch_size=batch,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=train_cfg["learning_rate"]); best=float("inf"); state=None; stale=0; history=[]
    for epoch in range(1,max_epochs+1):
        model.train(); total=0.; count=0
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True); loss=model.loss(x.to(device),y.to(device),epoch)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite {name} loss at epoch {epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),train_cfg["gradient_clip"]); optimizer.step()
            total+=float(loss.detach()); count+=1
        val_scores=_score_batches(model,p.validation.x,p.validation.y_next,batch,device)
        # Unsupervised early stopping uses normal-validation score mean when available.
        normal=val_scores[p.validation.labels==0]; criterion=float(normal.mean() if len(normal) else val_scores.mean())
        history.append({"epoch":epoch,"train_loss":total/max(count,1),"normal_validation_score":criterion})
        if criterion<best-1e-7: best=criterion; stale=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=train_cfg["patience"]: break
    model.load_state_dict(state); model.to(device)
    val_scores=_score_batches(model,p.validation.x,p.validation.y_next,batch,device); threshold,_=select_f1_threshold(p.validation.labels,val_scores)
    test_scores=_score_batches(model,p.test.x,p.test.y_next,batch,device); result=binary_metrics(p.test.labels,test_scores,threshold)
    result.update({"auroc":roc_auc(p.test.labels,test_scores),"auprc":average_precision(p.test.labels,test_scores),
                   "delay":detection_delay(p.test.labels,test_scores>threshold)})
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    torch.save({"name":name,"state":state,"threshold":threshold,"config":cfg,"features":features,"seed":seed},out/"checkpoint.pt")
    np.savez_compressed(out/"test_outputs.npz",system_scores=test_scores,labels=p.test.labels,
                        fault_types=p.test.fault_types.astype(str),end_indices=p.test.end_indices,alarm_threshold=np.asarray(threshold))
    payload={"method":name,"compatibility_implementation":True,"epochs":len(history),"threshold":threshold,"test":result}
    (out/"metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    return payload
