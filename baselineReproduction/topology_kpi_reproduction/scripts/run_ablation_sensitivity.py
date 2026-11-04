#!/usr/bin/env python3
"""Run proposed-model ablations and sensitivity grids from the manuscript.

This script changes one declared factor at a time, writes every resolved config,
and never uses the manuscript's reported table as computed output.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np

from topokpi.data import load_npz, save_npz, RawDataset
from topokpi.experiment import evaluate, load_config, train


def set_path(config: dict, dotted: str, value) -> None:
    target=config
    parts=dotted.split(".")
    for key in parts[:-1]: target=target[key]
    target[parts[-1]]=value


def variants(base: dict):
    yield "proposed", base, None
    c=copy.deepcopy(base); c["model"].update({"graph_layers":0,"topology_input_strength":0.0,"lambda_topology":0.0}); yield "without_topology",c,None
    c=copy.deepcopy(base); c["model"].update({"lambda_pred":0.0,"alpha":1.0}); yield "without_prediction",c,None
    c=copy.deepcopy(base); c["rca"]["gamma"]=1.0; yield "without_propagation",c,None
    yield "static_undirected_propagation",copy.deepcopy(base),"static"
    grids={"window_length":("data.window_length",[10,20,30]),"mask_ratio":("model.mask_ratio",[.10,.15,.20]),
           "propagation_depth":("rca.propagation_depth",[2,3,4]),"decay":("rca.decay",[.3,.5,.7,.9]),
           "gamma":("rca.gamma",[i/10 for i in range(11)]),"lambda_topology":("model.lambda_topology",[0,.01,.05,.1,.2,.5])}
    for group,(path,values) in grids.items():
        for value in values:
            c=copy.deepcopy(base); set_path(c,path,value); yield f"{group}_{value:g}",c,None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--config",default="configs/paper.json"); ap.add_argument("--output",required=True); ap.add_argument("--seed",type=int,default=17); ap.add_argument("--device"); ap.add_argument("--only",help="substring filter")
    args=ap.parse_args(); base=load_config(args.config); output=Path(args.output); output.mkdir(parents=True,exist_ok=True); raw=load_npz(args.data); results={}
    for name,cfg,topology_mode in variants(base):
        if args.only and args.only not in name: continue
        run=output/name; run.mkdir(parents=True,exist_ok=True); config_path=run/"resolved_config.json"; config_path.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
        data_path=args.data
        if topology_mode=="static":
            adapted=RawDataset(raw.x,raw.labels,raw.root_causes,raw.fault_types,raw.a_enc,raw.a_enc,raw.timestamps); data_path=str(run/"static_topology_data.npz"); save_npz(data_path,adapted)
        train(data_path,str(config_path),str(run),args.seed,args.device); results[name]=evaluate(str(run),data_path,args.device)
        (output/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__=="__main__": main()

