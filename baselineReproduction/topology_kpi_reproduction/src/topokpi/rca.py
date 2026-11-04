from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RCAResult:
    ranking: list[int]
    candidates: list[int]
    local: np.ndarray
    propagation: np.ndarray
    final: np.ndarray
    influence: np.ndarray
    contributions: np.ndarray | None
    explanation_edges: list[tuple[int, int, float]]


def row_normalize(w: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(w, float), 0.0).copy()
    np.fill_diagonal(w, 0.0)
    sums = w.sum(1, keepdims=True)
    return np.divide(w, sums, out=np.zeros_like(w), where=sums > 0)


def multi_hop_influence(w: np.ndarray, depth: int = 3, decay: float = .7) -> np.ndarray:
    p = row_normalize(w)
    total, power = np.zeros_like(p), p.copy()
    for hop in range(1, depth + 1):
        total += decay ** (hop - 1) * power
        power = power @ p
    np.fill_diagonal(total, 0.0)
    return total


def localize(node_scores: np.ndarray, w: np.ndarray, node_threshold: float,
             candidate_cap: int = 5, depth: int = 3, decay: float = .7,
             gamma: float = .6, kpi_scores: np.ndarray | None = None) -> RCAResult:
    scores = np.maximum(np.asarray(node_scores, float), 0.0)
    eligible = np.flatnonzero(scores > node_threshold)
    if not len(eligible):
        eligible = np.array([int(np.argmax(scores))])
    candidates = eligible[np.argsort(-scores[eligible], kind="stable")[:candidate_cap]]
    raw = multi_hop_influence(w, depth, decay)
    pi = np.zeros_like(raw)
    for i in candidates:
        denom = raw[i, candidates[candidates != i]].sum()
        if denom > 0:
            pi[i, candidates] = raw[i, candidates] / denom
            pi[i, i] = 0.0
    prop = pi @ scores
    local_norm = scores[candidates] / (scores[candidates].sum() + 1e-12)
    prop_norm = prop[candidates] / (prop[candidates].sum() + 1e-12)
    final_values = gamma * local_norm + (1 - gamma) * prop_norm
    full_final = np.zeros_like(scores)
    full_final[candidates] = final_values
    scored_ranking = candidates[np.argsort(-final_values, kind="stable")].tolist()
    # The algorithm scores only the post-alarm candidate set. For evaluation,
    # append every remaining node in descending local-anomaly order so the sole
    # injected root always has a well-defined rank beyond the A@5 cutoff.
    candidate_set = set(scored_ranking)
    remaining = [int(i) for i in np.argsort(-scores, kind="stable") if int(i) not in candidate_set]
    ranking = scored_ranking + remaining
    contributions = None
    if kpi_scores is not None:
        k = np.maximum(np.asarray(kpi_scores, float), 0.0)
        contributions = k / (k.sum(1, keepdims=True) + 1e-12)
    root = scored_ranking[0]
    edges = [(root, int(j), float(pi[root, j])) for j in candidates if j != root and pi[root, j] > 0]
    edges.sort(key=lambda item: (-item[2], item[1]))
    return RCAResult(ranking, candidates.tolist(), local_norm, prop_norm, full_final, pi, contributions, edges)


def personalized_pagerank(w: np.ndarray, anomaly: np.ndarray, damping: float = .85,
                          tolerance: float = 1e-6, iterations: int = 100) -> np.ndarray:
    p = row_normalize(w)
    prior = np.maximum(np.asarray(anomaly, float), 0)
    prior = prior / (prior.sum() + 1e-12)
    rank = prior.copy()
    for _ in range(iterations):
        nxt = damping * p.T @ rank + (1 - damping) * prior
        if np.abs(nxt - rank).sum() < tolerance:
            rank = nxt
            break
        rank = nxt
    return rank / (rank.sum() + 1e-12)
