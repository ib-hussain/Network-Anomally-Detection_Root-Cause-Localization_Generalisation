# Archive manifest

Version: 1.0.0

This archive accompanies the Scientific Reports manuscript titled *Topology-Aware Self-Supervised Multivariate KPI Anomaly Detection and Root Cause Node Localization for Network Operations*.

## Main contents

- `src/topokpi/`: proposed detector, topology-aware RCA, metrics, explainability, baseline interfaces, data handling, and experiment orchestration.
- `scripts/`: five-seed evaluation, ablation and sensitivity analyses, contamination experiments, external benchmark conversion, statistical comparisons, runtime benchmarking, and Mininet campaign utilities.
- `configs/`: manuscript, baseline, external benchmark, and revision settings.
- `tests/`: automated unit and compatibility tests.
- `data/reported/manuscript_results.json`: machine-readable values reported in the revised manuscript.
- `data/reported/five_seed/`: per-seed detection and RCA summaries, scale-stratified results, and paired-comparison inputs.
- `data/reported/secondary_analysis_deltas.csv`: within-protocol changes used in the revised ablation, RCA-weight, and contamination tables.

## Data boundary

The archive does not redistribute the Mininet KPI streams or the third-party DejaVu Dataset D. Their access routes are stated in the manuscript's Data Availability section. The generated demo dataset is only a software test fixture and is not used for any manuscript result.

## Reproducibility

The five aligned seeds are 17, 29, 43, 71, and 101. The evaluated topology sizes are 12, 24, and 48 nodes. Root-cause localization uses one deliberately injected ground-truth root per event; downstream affected nodes are excluded from the root-cause set.
