# Step 18 — TTHP delay-kernel implementation

Paper: Zhou et al. (2026), Section 4.2, Equation (4).

Implemented:
- Absolute-time-difference exponential kernel.
- Earlier, simultaneous and bounded later-event contributions.
- Exclusion of an event's own record.
- Same-device (zero-hop) candidate effects.
- Synthetic unit tests and Wireless data checks.

Not implemented:
- Full topology-aware Hawkes intensity.
- Modified likelihood and integral terms.
- EM optimization and causal graph search.
- Anomaly correlation graph weighting.
- GAT and LightGBM RCA training.

Scientific limitations:
- Physical Wireless topology is unavailable.
- The delay window and decay rate are exploratory.
- No DAG labels are used for selecting parameters.
- These outputs are not causal accuracy measurements.
- The complete TTHP reproduction remains pending.

Original exploratory THP runs remain unchanged.
