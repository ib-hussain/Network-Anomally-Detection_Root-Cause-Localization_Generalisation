# Step 19 — Experimental TTHP likelihood prototype

Implements:
- Binned alarm-event counts.
- Same-device exponential event effects.
- A configurable delay window.
- Source-to-target excitation coefficients.
- An unpenalized discrete Poisson likelihood.

Verified using synthetic numerical unit tests.

Limitations:
- Not a complete or trained TTHP implementation.
- No physical device topology.
- No EM optimization or causal graph search.
- No sparsity penalty.
- Same-bin and boundary handling need further validation.
- Quadratic implementation, limited to 200 time bins.
- Not suitable for paper-level benchmark scores.
