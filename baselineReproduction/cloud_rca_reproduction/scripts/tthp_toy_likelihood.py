"""Experimental discrete-time, same-device Hawkes likelihood.

Validates delay-window intensity and likelihood calculations.
NOT the trained, topology-aware TTHP model.
"""

import numpy as np


def intensity(counts, mu, alpha, decay,
              delay_bins, bin_width=1.0):
    """Return rates shaped [nodes, bins, event_types].

    alpha[source_type, target_type] is an excitation
    coefficient. Only same-device effects are modeled.
    """
    x = np.asarray(counts, dtype=float)
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    if x.ndim != 3:
        raise ValueError(
            "counts must be [nodes, bins, event_types]"
        )

    nodes, steps, types = x.shape

    if mu.shape != (types,) or alpha.shape != (types, types):
        raise ValueError("Parameter dimension mismatch")

    if ((x < 0).any() or
        not np.isfinite(x).all() or
        not np.isfinite(mu).all() or
        not np.isfinite(alpha).all()):
        raise ValueError("Invalid counts or parameters")

    if (mu <= 0).any() or (alpha < 0).any():
        raise ValueError(
            "mu must be positive and alpha nonnegative"
        )

    if (decay <= 0 or delay_bins < 0 or
        bin_width <= 0 or
        int(delay_bins) != delay_bins):
        raise ValueError("Invalid time parameters")

    # Intentionally small: this prototype is quadratic.
    if steps > 200:
        raise ValueError("Toy model allows at most 200 bins")

    effects = np.zeros_like(x)

    for t in range(steps):
        last_source = min(
            steps, t + int(delay_bins) + 1
        )

        for s in range(last_source):
            weight = np.exp(
                -decay * abs(t - s) * bin_width
            )
            effects[:, t, :] += weight * x[:, s, :]

    return mu[None, None, :] + effects @ alpha


def log_likelihood(counts, rate, bin_width=1.0):
    """Unpenalized discrete Poisson log-likelihood."""
    x = np.asarray(counts, dtype=float)
    rate = np.asarray(rate, dtype=float)

    if (x.shape != rate.shape or
        (rate <= 0).any() or
        not np.isfinite(rate).all() or
        bin_width <= 0):
        raise ValueError("Invalid count/rate inputs")

    return float(
        np.sum(x * np.log(rate) - bin_width * rate)
    )
