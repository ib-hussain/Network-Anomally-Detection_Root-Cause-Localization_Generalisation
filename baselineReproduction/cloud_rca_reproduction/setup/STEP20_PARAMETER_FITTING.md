# Step 20 — Hawkes parameter fitting

Implemented:
- Strictly past-only Hawkes event effects.
- Nonnegative excitation parameters.
- Positive background intensities.
- L-BFGS-B parameter optimization.
- Analytical gradient.
- Numerical gradient unit test.
- Chronological training/validation separation.
- Synthetic known-parameter experiment.

Not implemented:
- Paper's EM optimizer.
- Paper's delay-aware TTHP intensity.
- Topology-aware interactions.
- Causal graph edge selection or BIC graph search.
- Training with Wireless alarm data.
- GAT and LightGBM RCA.

This is a numerical parameter-estimation baseline,
not a reproduction of the published TTHP model.

The earlier Step 19 toy likelihood includes current-bin
observations in the effect calculation. It must not be
used directly for causal parameter fitting without
resolving same-bin self-contributions.
