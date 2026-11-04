#!/usr/bin/env python3
"""Evaluate the KPI-only RCA baselines from saved detector outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from topokpi.data import load_npz
from topokpi.experiment import event_representatives
from topokpi.metrics import rca_metrics
from topokpi.rca_baselines import rank_rca


METHODS=["OLS-residual","VAR(1)-residual","CauseInfer","MonitorRank","CloudRanger","MicroCause","TraceRCA (restricted)"]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",required=True); p.add_argument("--data",required=True); p.add_argument("--history",type=int,default=20); p.add_argument("--candidate-cap",type=int,default=5); args=p.parse_args()
    raw=load_npz(args.data); z=np.load(Path(args.run)/"test_outputs.npz"); system,node,kpi=z["system_scores"],z["node_scores"],z["kpi_scores"]; labels,roots,ends=z["labels"],z["root_causes"],z["end_indices"]; alarms=system>float(z["alarm_threshold"]); node_threshold=float(z["node_threshold"])
    events=event_representatives(labels,roots,system,alarms); results={}
    for method in METHODS:
        rankings=[]; truth=[]
        for q in events:
            lo=max(0,q-args.history+1); raw_time=int(ends[q]); topology=raw.w_rca[raw_time] if raw.w_rca.ndim==3 else raw.w_rca
            candidate=np.flatnonzero(node[q]>node_threshold)
            if not len(candidate): candidate=np.array([int(np.argmax(node[q]))])
            candidate=candidate[np.argsort(-node[q,candidate],kind="stable")[:args.candidate_cap]]
            ranked=rank_rca(method,node[lo:q+1],topology,kpi[lo:q+1]); candidate_set=set(candidate.tolist()); filtered=[i for i in ranked if i in candidate_set]
            rankings.append(filtered+[i for i in ranked if i not in candidate_set]); truth.append(int(roots[q]))
        results[method]=rca_metrics(rankings,truth) if truth else {}
    payload={"note":"KPI-only input-matched compatibility implementations; Nezha requires a separately trained supervised adapter.","events":len(events),"metrics":results}
    destination=Path(args.run)/"rca_baseline_metrics.json"; destination.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
