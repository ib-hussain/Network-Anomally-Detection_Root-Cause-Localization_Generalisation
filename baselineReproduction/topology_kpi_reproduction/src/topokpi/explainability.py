from __future__ import annotations
import numpy as np


def kpi_hit_rate(contributions: np.ndarray, roots: np.ndarray,
                 dominant_kpis: list[int | list[int]], k: int = 1) -> float:
    """Fraction of events where a known dominant KPI is among root-node top-k residual shares."""
    contributions=np.asarray(contributions,float); roots=np.asarray(roots,int); hits=[]
    for event,root in enumerate(roots):
        truth=dominant_kpis[event]; truth={int(truth)} if np.isscalar(truth) else {int(v) for v in truth}
        predicted=set(np.argsort(-contributions[event,root],kind="stable")[:k].tolist()); hits.append(bool(predicted & truth))
    return float(np.mean(hits)) if hits else float("nan")


def path_overlap(predicted_edges: list[tuple[int,int]], true_edges: list[tuple[int,int]]) -> dict[str,float]:
    predicted,true=set(map(tuple,predicted_edges)),set(map(tuple,true_edges)); intersection=len(predicted & true)
    precision=intersection/len(predicted) if predicted else 0.; recall=intersection/len(true) if true else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {"edge_precision":precision,"edge_recall":recall,"edge_f1":f1,"any_overlap":float(intersection>0)}


def explainability_metrics(contributions: np.ndarray, roots: np.ndarray, dominant_kpis: list,
                           predicted_paths: list[list[tuple[int,int]]], true_paths: list[list[tuple[int,int]]]) -> dict[str,float]:
    overlaps=[path_overlap(p,t) for p,t in zip(predicted_paths,true_paths)]
    result={"kpi_hit@1":kpi_hit_rate(contributions,roots,dominant_kpis,1),"kpi_hit@3":kpi_hit_rate(contributions,roots,dominant_kpis,3)}
    for key in ("edge_precision","edge_recall","edge_f1","any_overlap"):
        result["path_"+key]=float(np.mean([o[key] for o in overlaps])) if overlaps else float("nan")
    return result

