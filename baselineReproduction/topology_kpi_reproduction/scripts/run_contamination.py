#!/usr/bin/env python3
"""Construct and run the manuscript's window-level contamination conditions.

The abnormal pool must be a separate NPZ dataset and is never drawn from the
validation/test windows of the target dataset.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
from topokpi.data import load_npz, make_windows, prepare
from topokpi.experiment import evaluate, load_config, train


def pool_windows(pool, mean, std, window, stride):
    x=((pool.x-mean[None,None,:])/std[None,None,:]).astype(np.float32)
    windows=make_windows(x,pool.labels,pool.root_causes,pool.fault_types,0,len(x),window,stride,"overlap")
    keep=windows.labels==1
    return windows.x[keep],windows.y_next[keep]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--abnormal-pool",required=True); p.add_argument("--config",default="configs/paper.json"); p.add_argument("--output",required=True); p.add_argument("--seed",type=int,default=17); p.add_argument("--device"); args=p.parse_args()
    base=load_config(args.config); raw=load_npz(args.data); pool=load_npz(args.abnormal_pool); dc=base["data"]
    if raw.x.shape[1:]!=pool.x.shape[1:]: raise SystemExit("target data and abnormal pool must have identical node/KPI dimensions")
    prepared=prepare(raw,dc["window_length"],dc["stride"],tuple(dc["split"])); clean=np.flatnonzero(prepared.train.labels==0)
    abnormal_x,abnormal_y=pool_windows(pool,prepared.mean,prepared.std,dc["window_length"],dc["stride"]); output=Path(args.output); output.mkdir(parents=True,exist_ok=True); results={}
    for ratio in [0,.01,.03,.05,.10,.15]:
        rng=np.random.default_rng(args.seed); total=len(prepared.train.x); abnormal_count=int(round(total*ratio)); normal_count=total-abnormal_count
        if normal_count>len(clean) or abnormal_count>len(abnormal_x): raise SystemExit(f"insufficient clean/abnormal windows for ratio {ratio:g}")
        normal_idx=rng.choice(clean,normal_count,replace=False); abnormal_idx=rng.choice(len(abnormal_x),abnormal_count,replace=False) if abnormal_count else np.empty(0,int)
        x=np.concatenate((prepared.train.x[normal_idx],abnormal_x[abnormal_idx])); y=np.concatenate((prepared.train.y_next[normal_idx],abnormal_y[abnormal_idx])); order=rng.permutation(total); x,y=x[order],y[order]
        run=output/f"contamination_{ratio:g}"; run.mkdir(parents=True,exist_ok=True); override=run/"training_windows.npz"; np.savez_compressed(override,x=x,y_next=y)
        cfg=copy.deepcopy(base); cfg["data"]["training_windows_npz"]=str(override.resolve()); cfg_path=run/"resolved_config.json"; cfg_path.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
        train(args.data,str(cfg_path),str(run),args.seed,args.device); results[str(ratio)]=evaluate(str(run),args.data,args.device); (output/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__=="__main__": main()

