#!/usr/bin/env python3
"""Evaluate KPI contributions and propagation paths using event annotations JSON."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from topokpi.data import load_npz
from topokpi.explainability import explainability_metrics
from topokpi.rca import localize


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",required=True); p.add_argument("--data",required=True); p.add_argument("--annotations",required=True,help="JSON list aligned with saved detected events"); args=p.parse_args()
    out=np.load(Path(args.run)/"test_outputs.npz"); raw=load_npz(args.data); payload=json.loads(Path(args.annotations).read_text(encoding="utf-8")); annotations=payload.get("events",payload) if isinstance(payload,dict) else payload; indices=out["event_indices"]
    if isinstance(annotations,list) and len(annotations)!=len(indices):
        by_fault={item["fault_type"]:item for item in annotations if "fault_type" in item}; annotations=[by_fault[str(f)] for f in out["event_fault_types"]]
    if len(annotations)!=len(indices): raise SystemExit("annotations could not be aligned with detected events")
    contributions=[]; roots=[]; dominant=[]; predicted=[]; truth=[]
    for row,(q,ann) in enumerate(zip(indices,annotations)):
        raw_time=int(out["end_indices"][q]); w=raw.w_rca[raw_time] if raw.w_rca.ndim==3 else raw.w_rca
        result=localize(out["node_scores"][q],w,float(out["node_threshold"]),kpi_scores=out["kpi_scores"][q])
        contributions.append(result.contributions); roots.append(int(out["event_roots"][row])); dominant.append(ann["dominant_kpis"])
        predicted.append([(i,j) for i,j,_ in result.explanation_edges]); truth.append([tuple(edge) for edge in ann["true_path_edges"]])
    metrics=explainability_metrics(np.asarray(contributions),np.asarray(roots),dominant,predicted,truth); destination=Path(args.run)/"explainability_metrics.json"; destination.write_text(json.dumps(metrics,indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2))


if __name__=="__main__": main()
