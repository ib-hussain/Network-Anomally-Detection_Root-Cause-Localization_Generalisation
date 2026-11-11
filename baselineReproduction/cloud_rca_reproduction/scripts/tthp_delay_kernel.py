"""Reference event-level TTHP delay-kernel calculation.

Implements candidate event contributions for Equation (4)
in the same-device (zero-hop) setting.

This is NOT a complete TTHP training implementation.
"""

import numpy as np


def event_effects(events, delay, delta, n_types):
    """Calculate exponential effects from candidate events.

    Returns a matrix of shape [n_events, n_types].

    For each target event, include:
      - All strictly earlier events on its device.
      - Other simultaneous events on its device.
      - Later events up to the collection-delay interval.

    The event itself is excluded.
    """

    if delay < 0 or delta <= 0 or n_types <= 0:
        raise ValueError(
            "delay must be >= 0; delta and n_types must be > 0"
        )

    required = {"event", "timestamp", "node"}
    if not required.issubset(events.columns):
        raise ValueError(
            "Required columns: event, timestamp, node"
        )

    times = events["timestamp"].to_numpy(dtype=float)
    kinds = events["event"].to_numpy(dtype=int)
    nodes = events["node"].to_numpy()

    if not np.isfinite(times).all():
        raise ValueError("Invalid timestamps")

    if not ((0 <= kinds) & (kinds < n_types)).all():
        raise ValueError("Invalid event type IDs")

    result = np.zeros((len(events), n_types), dtype=float)

    for node in np.unique(nodes):
        positions = np.flatnonzero(nodes == node)
        node_times = times[positions]
        node_types = kinds[positions]

        for local_idx, global_idx in enumerate(positions):
            t = node_times[local_idx]

            # All past events, plus future events within delay.
            mask = node_times <= t + delay

            # Prevent an event from influencing itself.
            mask[local_idx] = False

            differences = np.abs(node_times[mask] - t)
            weights = np.exp(-delta * differences)

            np.add.at(
                result[global_idx],
                node_types[mask],
                weights
            )

    return result
