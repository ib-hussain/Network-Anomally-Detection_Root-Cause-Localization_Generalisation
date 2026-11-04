#!/usr/bin/env python3
"""Train/evaluate the explicitly restricted KPI+topology Nezha compatibility model."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from topokpi.data import load_npz,prepare
from topokpi.experiment import load_config
from topokpi.metrics import rca_metrics
from topokpi.rca_baselines import NezhaRestrictedCompat


def event_endpoints(labels,roots):
    return np.flatnonzero((labels==1)&np.r_[labels[1:]==0,True] & (roots>=0))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--detector-run",required=True); p.add_argument("--config",default="configs/paper.json"); p.add_argument("--output",required=True); p.add_argument("--seed",type=int,default=17); p.add_argument("--device"); args=p.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg=load_config(args.config); raw=load_npz(args.data); dc=cfg["data"]; prepared=prepare(raw,dc["window_length"],dc["stride"],tuple(dc["split"])); device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model=NezhaRestrictedCompat(raw.x.shape[-1]).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=cfg["training"]["learning_rate"]); topology=torch.tensor(raw.a_enc,dtype=torch.float32,device=device)
    train_idx=event_endpoints(prepared.train.labels,prepared.train.root_causes); val_idx=event_endpoints(prepared.validation.labels,prepared.validation.root_causes)
    if not len(train_idx): raise SystemExit("no labeled root-cause events in training split; restricted Nezha cannot be trained")
    best=float("inf"); state=None; stale=0; history=[]
    for epoch in range(1,cfg["training"]["max_epochs"]+1):
        model.train(); order=np.random.permutation(train_idx); losses=[]
        for start in range(0,len(order),cfg["training"]["batch_size"]):
            idx=order[start:start+cfg["training"]["batch_size"]]; x=torch.from_numpy(prepared.train.x[idx]).to(device); target=torch.from_numpy(prepared.train.root_causes[idx].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True); logits=model(x,topology); loss=torch.nn.functional.cross_entropy(logits,target); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["gradient_clip"]); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            if len(val_idx): val=float(torch.nn.functional.cross_entropy(model(torch.from_numpy(prepared.validation.x[val_idx]).to(device),topology),torch.from_numpy(prepared.validation.root_causes[val_idx].astype(np.int64)).to(device)))
            else: val=float(np.mean(losses))
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":val})
        if val<best-1e-7: best=val; stale=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=cfg["training"]["patience"]: break
    model.load_state_dict(state); model.to(device).eval(); detector=np.load(Path(args.detector_run)/"test_outputs.npz"); rankings=[]; truth=[]
    with torch.no_grad():
        for event_row,q in enumerate(detector["event_indices"]):
            logits=model(torch.from_numpy(prepared.test.x[[q]]).to(device),topology)[0].cpu().numpy(); candidates=np.flatnonzero(detector["node_scores"][q]>float(detector["node_threshold"]));
            if not len(candidates): candidates=np.array([int(np.argmax(detector["node_scores"][q]))])
            candidates=candidates[np.argsort(-detector["node_scores"][q,candidates],kind="stable")[:cfg["rca"]["candidate_cap"]]]; scored=candidates[np.argsort(-logits[candidates],kind="stable")].tolist(); candidate_set=set(scored); remaining=[int(i) for i in np.argsort(-logits,kind="stable") if int(i) not in candidate_set]; rankings.append(scored+remaining); truth.append(int(detector["event_roots"][event_row]))
    payload={"method":"Nezha-inspired restricted adaptation","native_multimodal_reproduction":False,"training_events":len(train_idx),"validation_events":len(val_idx),"test_events":len(truth),"metrics":rca_metrics(rankings,truth) if truth else {}}
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True); torch.save({"state":state,"config":cfg},output/"checkpoint.pt"); (output/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8"); (output/"metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
