from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


KPI_NAMES = np.array([
    "cpu", "memory", "inbound_traffic", "outbound_traffic",
    "rtt", "packet_loss", "queue_length", "interface_errors",
])
FAULT_TYPES = np.array([
    "cpu_overload", "bandwidth_bottleneck", "packet_loss",
    "link_delay", "interface_failure", "routing_instability",
])


@dataclass
class RawDataset:
    x: np.ndarray
    labels: np.ndarray
    root_causes: np.ndarray
    fault_types: np.ndarray
    a_enc: np.ndarray
    w_rca: np.ndarray
    timestamps: np.ndarray

    def validate(self) -> None:
        if self.x.ndim != 3:
            raise ValueError("x must have shape [time,nodes,kpis]")
        t, n, _ = self.x.shape
        for name, value in (("labels", self.labels), ("root_causes", self.root_causes), ("fault_types", self.fault_types), ("timestamps", self.timestamps)):
            if len(value) != t:
                raise ValueError(f"{name} length {len(value)} does not match time {t}")
        if self.a_enc.shape != (n, n):
            raise ValueError("a_enc must have shape [nodes,nodes]")
        if self.w_rca.shape not in ((n, n), (t, n, n)):
            raise ValueError("w_rca must be [nodes,nodes] or [time,nodes,nodes]")
        if not np.isfinite(self.x).all():
            raise ValueError("x contains NaN or infinite values")
        if ((self.labels != 0) & (self.labels != 1)).any():
            raise ValueError("labels must be binary")


@dataclass
class Windows:
    x: np.ndarray
    y_next: np.ndarray
    labels: np.ndarray
    root_causes: np.ndarray
    fault_types: np.ndarray
    end_indices: np.ndarray


@dataclass
class PreparedData:
    train: Windows
    validation: Windows
    test: Windows
    mean: np.ndarray
    std: np.ndarray
    a_enc: np.ndarray
    w_rca: np.ndarray


def save_npz(path: str | Path, data: RawDataset) -> None:
    data.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=data.x.astype(np.float32), labels=data.labels.astype(np.int8),
                        root_causes=data.root_causes.astype(np.int32), fault_types=data.fault_types.astype(str),
                        a_enc=data.a_enc.astype(np.float32), w_rca=data.w_rca.astype(np.float32), timestamps=data.timestamps)


def load_npz(path: str | Path) -> RawDataset:
    with np.load(path, allow_pickle=False) as z:
        required = {"x", "labels", "root_causes", "fault_types", "a_enc", "w_rca"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"Missing NPZ fields: {sorted(missing)}")
        x = z["x"]
        data = RawDataset(x=x, labels=z["labels"], root_causes=z["root_causes"],
                          fault_types=z["fault_types"], a_enc=z["a_enc"], w_rca=z["w_rca"],
                          timestamps=z["timestamps"] if "timestamps" in z.files else np.arange(len(x)))
    data.validate()
    return data


def _ring_with_chords(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for step in (1, 3):
            j = (i + step) % n
            a[i, j] = a[j, i] = 1.0
    return a


def generate_demo(nodes: int = 12, steps: int = 1200, seed: int = 17, event_start_fraction: float = .62) -> RawDataset:
    """Create a deterministic schema test fixture, not experimental evidence."""
    if nodes < 6 or steps < 300:
        raise ValueError("demo requires at least 6 nodes and 300 steps")
    rng = np.random.default_rng(seed)
    d = len(KPI_NAMES)
    a = _ring_with_chords(nodes)
    w = np.zeros_like(a)
    for i in range(nodes):
        w[i, (i + 1) % nodes] = 1
        w[i, (i + 3) % nodes] = 1
    x = np.zeros((steps, nodes, d), dtype=np.float32)
    seasonal = np.stack([np.sin(np.arange(steps) / p) for p in (17, 29, 41, 23, 37, 31, 19, 47)], axis=-1)
    node_scale = rng.uniform(0.8, 1.2, size=(nodes, d))
    x[0] = rng.normal(0, .2, size=(nodes, d))
    for t in range(1, steps):
        neighbor = (a @ x[t - 1]) / np.maximum(a.sum(1, keepdims=True), 1)
        x[t] = 0.72 * x[t - 1] + 0.08 * neighbor + 0.16 * seasonal[t] * node_scale + rng.normal(0, .08, size=(nodes, d))
    labels = np.zeros(steps, dtype=np.int8)
    roots = np.full(steps, -1, dtype=np.int32)
    faults = np.full(steps, "normal", dtype="<U32")
    # Six events in the final 40%, so chronological training remains predominantly normal.
    if not .1 <= event_start_fraction <= .8: raise ValueError("event_start_fraction must be in [0.1,0.8]")
    starts = np.linspace(int(steps * event_start_fraction), steps - 55, 6, dtype=int)
    affected_kpis = ([0, 6, 7], [2, 3, 6], [5, 4], [4, 5], [7, 3], [2, 3, 4])
    for event, (start, name, dims) in enumerate(zip(starts, FAULT_TYPES, affected_kpis)):
        root = (2 * event + 1) % nodes
        duration = 24
        for hop, node in enumerate((root, (root + 1) % nodes, (root + 2) % nodes)):
            lo, hi = start + hop * 2, min(start + duration, steps)
            magnitude = 2.7 * (0.68 ** hop)
            x[lo:hi, node, dims] += magnitude
            labels[lo:hi] = 1
            roots[lo:hi] = root
            faults[lo:hi] = name
    return RawDataset(x, labels, roots, faults, a, w, np.arange(steps) * 30)


def make_windows(x: np.ndarray, labels: np.ndarray, roots: np.ndarray, faults: np.ndarray,
                 start: int, stop: int, length: int, stride: int = 1,
                 label_mode: Literal["endpoint", "overlap"] = "endpoint") -> Windows:
    # A one-step target is required, so the last possible endpoint is stop-2.
    ends = np.arange(start + length - 1, stop - 1, stride, dtype=int)
    if len(ends) == 0:
        shape = (0, length, x.shape[1], x.shape[2])
        return Windows(np.empty(shape, np.float32), np.empty((0, x.shape[1], x.shape[2]), np.float32),
                       np.empty(0, np.int8), np.empty(0, np.int32), np.empty(0, faults.dtype), ends)
    windows = np.stack([x[e-length+1:e+1] for e in ends]).astype(np.float32)
    if label_mode == "endpoint":
        ys, rs, fs = labels[ends], roots[ends], faults[ends]
    else:
        ys = np.array([labels[e-length+1:e+1].max() for e in ends], dtype=np.int8)
        rs, fs = roots[ends].copy(), faults[ends].copy()
        for q, e in enumerate(ends):
            if ys[q] and rs[q] < 0:
                hit = np.flatnonzero(labels[e-length+1:e+1])[-1] + e - length + 1
                rs[q], fs[q] = roots[hit], faults[hit]
    return Windows(windows, x[ends + 1].astype(np.float32), ys.astype(np.int8), rs.astype(np.int32), fs, ends)


def prepare(data: RawDataset, window: int = 20, stride: int = 1,
            split: tuple[float, float, float] = (.60, .15, .25), label_mode: str = "endpoint") -> PreparedData:
    data.validate()
    if not np.isclose(sum(split), 1.0):
        raise ValueError("split fractions must sum to one")
    t = len(data.x)
    train_stop, val_stop = int(t * split[0]), int(t * (split[0] + split[1]))
    mean = data.x[:train_stop].mean(axis=(0, 1), keepdims=True)
    std = data.x[:train_stop].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = ((data.x - mean) / std).astype(np.float32)
    train = make_windows(x, data.labels, data.root_causes, data.fault_types, 0, train_stop, window, stride, label_mode)
    validation = make_windows(x, data.labels, data.root_causes, data.fault_types, train_stop, val_stop, window, stride, label_mode)
    test = make_windows(x, data.labels, data.root_causes, data.fault_types, val_stop, t, window, stride, label_mode)
    return PreparedData(train, validation, test, mean.squeeze((0, 1)), std.squeeze((0, 1)), data.a_enc, data.w_rca)


def replace_with_contamination(clean_windows: np.ndarray, abnormal_pool: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Replace clean windows with a disjoint abnormal pool while preserving size."""
    if not 0 <= ratio < 1:
        raise ValueError("ratio must be in [0,1)")
    count = int(round(len(clean_windows) * ratio))
    if count > len(abnormal_pool):
        raise ValueError("abnormal pool is too small for requested replacement")
    rng = np.random.default_rng(seed)
    result = clean_windows.copy()
    if count:
        dst = rng.choice(len(result), count, replace=False)
        src = rng.choice(len(abnormal_pool), count, replace=False)
        result[dst] = abnormal_pool[src]
    return result
