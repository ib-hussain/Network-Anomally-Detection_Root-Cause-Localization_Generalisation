"""Inspect TTHP delay-window candidates.

Diagnostics only. Does not learn a causal graph.
Counts same-device candidate pairs without using
unknown physical topology or ground-truth DAG labels.
"""

import numpy as np
import pandas as pd

DATA = "data/processed/wireless_thp_events.csv"


def candidate_counts(times, delay):
    times = np.sort(np.asarray(times, dtype=np.int64))

    left = np.searchsorted(times, times, side="left")
    right = np.searchsorted(times, times, side="right")

    # Other events at the exact same timestamp.
    same_time = int(np.sum(right - left - 1))

    # Strictly later events within the delay interval.
    future_end = np.searchsorted(
        times, times + delay, side="right"
    )
    delayed = int(np.sum(future_end - right))

    return same_time, delayed


def main():
    # Small controlled test to verify window boundaries.
    example = np.array([10, 10, 12, 25])

    assert candidate_counts(example, 0) == (2, 0)
    assert candidate_counts(example, 2) == (2, 2)

    print("Synthetic boundary tests: PASSED")

    data = pd.read_csv(DATA)

    print("\n=== WIRELESS DELAY INSPECTION ===")
    print("Events:", len(data))
    print("Devices:", data["node"].nunique())
    print("Alarm types:", data["event"].nunique())

    # These are exploratory windows in raw timestamp units.
    # They are NOT verified paper hyperparameters.
    for delay in (0, 1, 10, 60):
        same_total = 0
        delayed_total = 0

        for _, device in data.groupby("node"):
            same, later = candidate_counts(
                device["timestamp"].to_numpy(), delay
            )
            same_total += same
            delayed_total += later

        print(f"\nDelay window: {delay}")
        print("Same-time candidate pairs:", same_total)
        print("Later candidate pairs:", delayed_total)
        print("Total:", same_total + delayed_total)

    print("\nThese counts are NOT learned causal edges.")
    print("Only same-device events were examined.")
    print("Physical topology is still unavailable.")


if __name__ == "__main__":
    main()
