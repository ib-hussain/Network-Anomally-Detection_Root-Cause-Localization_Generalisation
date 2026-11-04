from __future__ import annotations
import numpy as np


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = s > threshold
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    values = np.unique(np.asarray(scores, float))
    if not len(values):
        raise ValueError("empty scores")
    candidates = np.r_[np.nextafter(values[0], -np.inf), (values[:-1] + values[1:]) / 2, values[-1]]
    best = (-1.0, float(candidates[0]), {})
    for threshold in candidates:
        result = binary_metrics(labels, scores, float(threshold))
        key = (result["f1"], -float(threshold))
        if key > (best[0], -best[1]):
            best = (result["f1"], float(threshold), result)
    return best[1], best[2]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        return float("nan")
    # Pairwise Mann-Whitney statistic; deterministic and dependency-free.
    return float(np.mean((s[pos, None] > s[neg]).astype(float) + .5 * (s[pos, None] == s[neg])))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, int)
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    ranked = y[order]
    total = ranked.sum()
    if total == 0:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / total)


def rca_metrics(rankings: list[list[int]], true_roots: list[int | list[int]]) -> dict[str, float]:
    """Single-root event-level A@k, MRR, and Mean Rank.

    Every event must have exactly one injected ground-truth root. ``rankings`` must
    contain that root; callers should append non-candidate nodes to the scored
    candidate list so MRR and Mean Rank are calculated from a complete node ranking.
    """
    if len(rankings) != len(true_roots):
        raise ValueError("rankings and true_roots must have the same length")
    if not rankings:
        raise ValueError("at least one RCA event is required")

    ranks = []
    for ranked, root in zip(rankings, true_roots):
        if np.isscalar(root):
            root_id = int(root)
        else:
            root_list = [int(value) for value in root]
            if len(root_list) != 1:
                raise ValueError("single-root RCA requires exactly one ground-truth root per event")
            root_id = root_list[0]
        if len(set(ranked)) != len(ranked):
            raise ValueError("RCA rankings must not contain duplicate node identifiers")
        if root_id not in ranked:
            raise ValueError("the ground-truth root is missing from an RCA ranking")
        ranks.append(ranked.index(root_id) + 1)

    rank = np.asarray(ranks, dtype=float)
    return {
        "a@1": float(np.mean(rank <= 1)),
        "a@3": float(np.mean(rank <= 3)),
        "a@5": float(np.mean(rank <= 5)),
        "mrr": float(np.mean(1.0 / rank)),
        "mean_rank": float(np.mean(rank)),
    }


def detection_delay(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels, predictions = np.asarray(labels, int), np.asarray(predictions, bool)
    starts = np.flatnonzero((labels == 1) & np.r_[True, labels[:-1] == 0])
    delays = []
    for start in starts:
        end_hits = np.flatnonzero(labels[start:] == 0)
        end = start + (end_hits[0] if len(end_hits) else len(labels) - start)
        hits = np.flatnonzero(predictions[start:end])
        delays.append(int(hits[0]) if len(hits) else end - start)
    return float(np.mean(delays)) if delays else float("nan")
