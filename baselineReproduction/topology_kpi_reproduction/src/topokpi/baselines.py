from __future__ import annotations
import numpy as np


class LinearResidualBaseline:
    """OLS (per-variable AR(1)) or joint VAR(1), with train-only residual scaling."""
    def __init__(self, joint: bool = False, ridge: float = 1e-6):
        self.joint, self.ridge = joint, ridge
        self.coef = self.scale = None

    def fit(self, x: np.ndarray) -> "LinearResidualBaseline":
        flat = np.asarray(x, float).reshape(len(x), -1)
        design = np.c_[np.ones(len(flat) - 1), flat[:-1]]
        target = flat[1:]
        if self.joint:
            gram = design.T @ design + self.ridge * np.eye(design.shape[1])
            self.coef = np.linalg.solve(gram, design.T @ target)
        else:
            # intercept and own lag for each variable
            count = flat.shape[1]
            self.coef = np.empty((2, count))
            for j in range(count):
                self.coef[:, j], *_ = np.linalg.lstsq(design[:, [0, j + 1]], target[:, j], rcond=None)
        residual = self.residuals(x)
        self.scale = np.maximum(np.std(residual, axis=0), 1e-6)
        return self

    def predict(self, x_previous: np.ndarray) -> np.ndarray:
        flat = np.asarray(x_previous, float).reshape(len(x_previous), -1)
        if self.joint:
            return np.c_[np.ones(len(flat)), flat] @ self.coef
        return self.coef[0] + flat * self.coef[1]

    def residuals(self, x: np.ndarray) -> np.ndarray:
        flat = np.asarray(x, float).reshape(len(x), -1)
        return np.abs(flat[1:] - self.predict(x[:-1]))

    def score(self, x: np.ndarray, nodes: int, kpis: int) -> tuple[np.ndarray, np.ndarray]:
        residual = self.residuals(x)
        scale = self.scale if self.scale is not None else np.maximum(residual.std(0), 1e-6)
        normalized = (residual / scale).reshape(len(residual), nodes, kpis)
        node = normalized.max(-1)
        return node.max(-1), node


def method_from_name(name: str) -> LinearResidualBaseline:
    if name == "OLS":
        return LinearResidualBaseline(joint=False)
    if name == "VAR(1)":
        return LinearResidualBaseline(joint=True)
    raise ValueError(f"Executable built-in baseline is unavailable for {name!r}; use its official adapter specified in configs/baselines.json")

