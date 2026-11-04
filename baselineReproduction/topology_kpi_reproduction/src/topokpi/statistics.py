from __future__ import annotations
import numpy as np
from .metrics import binary_metrics


def paired_bootstrap_detection(labels: np.ndarray, scores_a: np.ndarray, threshold_a: float,
                               scores_b: np.ndarray, threshold_b: float, samples: int = 5000,
                               seed: int = 17) -> dict[str,float | list[float]]:
    y=np.asarray(labels,int); a=np.asarray(scores_a)>threshold_a; b=np.asarray(scores_b)>threshold_b; rng=np.random.default_rng(seed); deltas=[]
    for _ in range(samples):
        idx=rng.integers(0,len(y),len(y)); fa=binary_metrics(y[idx],a[idx],.5)["f1"]; fb=binary_metrics(y[idx],b[idx],.5)["f1"]; deltas.append(fa-fb)
    values=np.asarray(deltas); observed=binary_metrics(y,a,.5)["f1"]-binary_metrics(y,b,.5)["f1"]
    return {"observed_f1_difference":observed,"bootstrap_mean":float(values.mean()),"ci95":np.quantile(values,[.025,.975]).tolist(),"two_sided_p":float(min(1.,2*min(np.mean(values<=0),np.mean(values>=0))))}


def paired_bootstrap_hits(hit_a: np.ndarray, hit_b: np.ndarray, samples: int = 5000, seed: int = 17) -> dict:
    a,b=np.asarray(hit_a,float),np.asarray(hit_b,float)
    if a.shape!=b.shape: raise ValueError("paired hit arrays must have equal shape")
    rng=np.random.default_rng(seed); values=np.empty(samples)
    for i in range(samples):
        idx=rng.integers(0,len(a),len(a)); values[i]=(a[idx]-b[idx]).mean()
    return {"observed_a1_difference":float((a-b).mean()),"ci95":np.quantile(values,[.025,.975]).tolist(),"two_sided_p":float(min(1.,2*min(np.mean(values<=0),np.mean(values>=0))))}

