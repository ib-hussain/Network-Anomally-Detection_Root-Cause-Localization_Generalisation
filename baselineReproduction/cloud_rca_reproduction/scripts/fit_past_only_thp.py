"""Numerical Hawkes parameter-fitting experiment.

Strictly past-only, same-device, discrete-time model.
Uses L-BFGS-B, NOT the paper's EM algorithm.
Does NOT implement TTHP or topology-aware learning.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

BETA = 0.6


def past_effects(counts, decay=BETA):
    x = np.asarray(counts, dtype=float)

    if (
        x.ndim != 3
        or not np.isfinite(x).all()
        or np.any(x < 0)
    ):
        raise ValueError("Invalid count tensor")

    h = np.zeros_like(x)
    r = np.exp(-decay)

    for t in range(1, x.shape[1]):
        h[:, t, :] = r * (
            h[:, t - 1, :] + x[:, t - 1, :]
        )

    return h


def objective_and_grad(params, x, features, penalty=0.02):
    types = x.shape[-1]

    mu = params[:types]
    alpha = params[types:].reshape(types, types)

    # alpha[source_type, target_type]
    rate = mu[None, None, :] + features @ alpha

    nll = (
        rate - x * np.log(rate) + gammaln(x + 1)
    ).sum() + penalty * (alpha ** 2).sum()

    derivative = 1.0 - x / rate

    grad_mu = derivative.sum(axis=(0, 1))
    grad_alpha = np.einsum(
        "ntk,ntj->kj", features, derivative
    ) + 2 * penalty * alpha

    gradient = np.concatenate(
        (grad_mu, grad_alpha.ravel())
    )

    return float(nll), gradient


def build_fixture(seed=7, devices=4, steps=400):
    rng = np.random.default_rng(seed)

    # Known synthetic source-type 0 -> target-type 1.
    mu = np.array([0.3, 0.12])
    alpha = np.array([
        [0.0, 0.28],
        [0.0, 0.0],
    ])

    x = np.zeros((devices, steps, 2))
    history = np.zeros((devices, 2))
    r = np.exp(-BETA)

    for t in range(steps):
        rate = mu + history @ alpha
        x[:, t, :] = rng.poisson(rate)
        history = r * (history + x[:, t, :])

    return x


def main():
    x = build_fixture()
    cutoff = int(x.shape[1] * 0.75)
    features = past_effects(x)

    x_train = x[:, :cutoff, :]
    h_train = features[:, :cutoff, :]

    n_types = x.shape[-1]

    initial = np.r_[
        np.maximum(x_train.mean(axis=(0, 1)), 0.05),
        np.zeros(n_types * n_types),
    ]

    initial_nll, _ = objective_and_grad(
        initial, x_train, h_train
    )

    result = minimize(
        objective_and_grad,
        initial,
        args=(x_train, h_train),
        method="L-BFGS-B",
        jac=True,
        bounds=(
            [(1e-6, None)] * n_types
            + [(0, None)] * (n_types * n_types)
        ),
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(result.message)

    fitted_nll, _ = objective_and_grad(
        result.x, x_train, h_train
    )

    alpha = result.x[n_types:].reshape(
        n_types, n_types
    )

    validation_nll, _ = objective_and_grad(
        result.x,
        x[:, cutoff:, :],
        features[:, cutoff:, :],
        penalty=0,
    )

    print("=== PARAMETER FITTING ===")
    print("Optimizer converged:", result.success)
    print("Initial training NLL:", round(initial_nll, 3))
    print("Fitted training NLL:", round(fitted_nll, 3))
    print("Estimated base rates:", result.x[:n_types])
    print("Estimated alpha[source, target]:")
    print(alpha)
    print(
        "Validation NLL per observation:",
        round(validation_nll / x[:, cutoff:, :].size, 4)
    )

    assert fitted_nll < initial_nll
    assert (result.x >= 0).all()

    print("Synthetic fitting checks: PASSED")


if __name__ == "__main__":
    main()
