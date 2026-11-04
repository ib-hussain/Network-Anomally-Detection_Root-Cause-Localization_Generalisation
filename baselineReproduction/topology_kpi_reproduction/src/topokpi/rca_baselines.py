"""Input-matched RCA baseline implementations for KPI + topology experiments."""
from __future__ import annotations
import numpy as np
from .rca import personalized_pagerank, row_normalize


def _lagged_influence(series: np.ndarray, max_lag: int = 3) -> np.ndarray:
    """Maximum positive lead/lag correlation: edge i->j when i leads j."""
    x = np.asarray(series, float)
    if x.ndim != 2: raise ValueError("series must be [time,nodes]")
    n = x.shape[1]; influence = np.zeros((n, n))
    for lag in range(1, min(max_lag, len(x)-2) + 1):
        source, target = x[:-lag], x[lag:]
        source = (source-source.mean(0))/np.maximum(source.std(0),1e-8)
        target = (target-target.mean(0))/np.maximum(target.std(0),1e-8)
        corr = source.T @ target / max(len(source)-1,1)
        influence = np.maximum(influence, np.maximum(corr, 0))
    np.fill_diagonal(influence, 0)
    return influence


def _topology_filter(graph: np.ndarray, topology: np.ndarray) -> np.ndarray:
    return np.asarray(graph, float) * (np.asarray(topology) > 0)


def residual_rank(node_scores: np.ndarray, candidates: np.ndarray | None = None) -> list[int]:
    ids = np.arange(len(node_scores)) if candidates is None else np.asarray(candidates, int)
    return ids[np.argsort(-np.asarray(node_scores)[ids], kind="stable")].tolist()


def causeinfer(series: np.ndarray, topology: np.ndarray, max_lag: int = 3) -> np.ndarray:
    graph = _topology_filter(_lagged_influence(series, max_lag), topology)
    # Upstream nodes are supported when they exert influence on abnormal neighbors.
    evidence = np.maximum(np.asarray(series)[-1], 0)
    return graph @ evidence


def monitorrank(node_scores: np.ndarray, topology: np.ndarray, damping: float = .85,
                tolerance: float = 1e-6, iterations: int = 100) -> np.ndarray:
    return personalized_pagerank(topology, node_scores, damping, tolerance, iterations)


def cloudranger(series: np.ndarray, topology: np.ndarray, max_lag: int = 3,
                restart: float = .15, tolerance: float = 1e-6, iterations: int = 100) -> np.ndarray:
    graph = _topology_filter(_lagged_influence(series, max_lag), topology)
    # Reverse walk favors upstream causes while restart retains local anomaly evidence.
    return personalized_pagerank(graph.T, np.maximum(series[-1], 0), 1-restart, tolerance, iterations)


def microcause(metric_series: np.ndarray, topology: np.ndarray, max_lag: int = 3) -> np.ndarray:
    x = np.asarray(metric_series, float)
    if x.ndim != 3: raise ValueError("metric_series must be [time,nodes,kpis]")
    n, d = x.shape[1:]; metric_scores = np.zeros((n, d))
    node_summary = np.max(np.abs(x), axis=2)
    graph = _topology_filter(_lagged_influence(node_summary, max_lag), topology)
    downstream = graph @ np.max(np.abs(x[-1]), axis=1)
    for k in range(d): metric_scores[:, k] = np.abs(x[-1, :, k]) * (1 + downstream)
    return metric_scores.max(1)


def tracerca_restricted(series: np.ndarray, topology: np.ndarray, max_lag: int = 3,
                        restart: float = .15, iterations: int = 100) -> np.ndarray:
    influence = _lagged_influence(np.abs(series), max_lag)
    impact = _topology_filter(influence + influence.T, topology)
    return personalized_pagerank(impact.T, np.maximum(series[-1], 0), 1-restart, 1e-6, iterations)


def rank_rca(name: str, node_series: np.ndarray, topology: np.ndarray,
             metric_series: np.ndarray | None = None) -> list[int]:
    if name in ("OLS-residual", "VAR(1)-residual"): scores = node_series[-1]
    elif name == "CauseInfer": scores = causeinfer(node_series, topology)
    elif name == "MonitorRank": scores = monitorrank(node_series[-1], topology)
    elif name == "CloudRanger": scores = cloudranger(node_series, topology)
    elif name == "MicroCause":
        if metric_series is None: raise ValueError("MicroCause requires metric_series")
        scores = microcause(metric_series, topology)
    elif name.startswith("TraceRCA"): scores = tracerca_restricted(node_series, topology)
    else: raise ValueError(f"No executable KPI-only RCA baseline for {name!r}")
    return np.argsort(-scores, kind="stable").tolist()


class NezhaRestrictedCompat:
    """Small supervised KPI/topology attention ranker for the restricted Nezha row.

    This class intentionally requires root labels for fitting and must not be trained
    on test events. It does not reproduce Nezha's unavailable multimodal inputs.
    """
    def __new__(cls, kpis: int, hidden: int = 64, heads: int = 4, layers: int = 2, dropout: float = .1):
        import torch
        from torch import nn
        class Module(nn.Module):
            def __init__(self):
                super().__init__(); self.inp=nn.Linear(kpis,hidden)
                self.layers=nn.ModuleList([nn.MultiheadAttention(hidden,heads,dropout,batch_first=True) for _ in range(layers)])
                self.out=nn.Linear(hidden,1)
            def forward(self,x,topology):
                # x [B,T,N,D]; attention mask restricts non-topological node pairs.
                h=self.inp(x.mean(1)); mask=torch.as_tensor(topology<=0,device=h.device); mask.fill_diagonal_(False)
                for layer in self.layers:
                    z,_=layer(h,h,h,attn_mask=mask); h=h+z
                return self.out(h).squeeze(-1)
        return Module()

