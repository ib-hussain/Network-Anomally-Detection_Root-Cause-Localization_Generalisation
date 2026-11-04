# Topology-aware KPI anomaly detection and root-cause localization

This archive accompanies the Scientific Reports manuscript *Topology-Aware Self-Supervised Multivariate KPI Anomaly Detection and Root Cause Node Localization for Network Operations*.

It contains the proposed self-supervised detector, directed topology-aware RCA procedure, baseline evaluation interfaces, experiment configurations, statistical utilities, and automated tests. The submitted experimental configuration uses shared model weights across topology sizes and contains approximately 0.42 million trainable parameters. Files under `data/reported/` record the numerical results reported in the manuscript and are not used as model inputs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick test

```bash
topokpi generate-demo --output demo.npz --nodes 12 --steps 1200
topokpi train --data demo.npz --config configs/paper.json --output run
topokpi evaluate --run run --data demo.npz
```

The demo dataset is synthetic and only verifies that the pipeline runs. It must not be presented as a reproduction of the paper's numerical results.

## Data format

Input is a compressed NumPy archive containing:

- `x`: `[time, nodes, kpis]`
- `labels`: binary `[time]`
- `root_causes`: node index or `-1`, `[time]`
- `fault_types`: `[time]`
- `a_enc`: encoder adjacency `[nodes,nodes]`
- `w_rca`: directed RCA topology `[nodes,nodes]` or `[time,nodes,nodes]`
- `timestamps`: `[time]`

Normalization uses training data only. The default chronological split is 60%/15%/25%, the window length is 20 and the stride is 1.

## Experiments

```bash
# Five seeds and confidence intervals
PYTHONPATH=src python scripts/run_five_seeds.py --data DATA.npz --output runs/five_seeds

# Ablation and sensitivity grids
PYTHONPATH=src python scripts/run_ablation_sensitivity.py --data DATA.npz --output runs/sweeps

# Training contamination: 0%, 1%, 3%, 5%, 10%, 15%
PYTHONPATH=src python scripts/run_contamination.py --data DATA.npz --abnormal-pool POOL.npz --output runs/contamination

# Revision Tables 13--16: 12/24/48 nodes x five aligned seeds
cp configs/revision_manifest.example.json configs/revision_manifest.json
# Edit the six NPZ paths in the copied manifest, then run:
PYTHONPATH=src python scripts/run_revision_tables_13_16.py --manifest configs/revision_manifest.json --output runs/revision_tables_13_16

# Detection baselines
topokpi baseline --name OLS --data DATA.npz
topokpi baseline --name 'VAR(1)' --data DATA.npz
topokpi deep-baseline --name OmniAnomaly --data DATA.npz --output runs/omni

# RCA baselines and restricted Nezha adaptation
PYTHONPATH=src python scripts/run_rca_baselines.py --run runs/proposed --data DATA.npz
PYTHONPATH=src python scripts/run_nezha_restricted.py --data DATA.npz --detector-run runs/proposed --output runs/nezha

# Explainability, aggregation, statistics and runtime
PYTHONPATH=src python scripts/evaluate_explanations.py --run runs/proposed --data DATA.npz --annotations annotations.json
PYTHONPATH=src python scripts/aggregate_campaigns.py --manifest runs.json --output runs/aggregate
PYTHONPATH=src python scripts/compare_runs.py --a runs/proposed --b runs/baseline
PYTHONPATH=src python scripts/benchmark_runtime.py --nodes 48
```

## Mininet testbed

```bash
# Review the generated campaign without Mininet
PYTHONPATH=src python scripts/mininet_campaign.py --nodes 12 --steps 1200 --interval 30 --output mininet_12.npz --dry-run

# Actual collection requires Linux, root and Mininet
sudo env PYTHONPATH=src python scripts/mininet_campaign.py --nodes 12 --steps 1200 --interval 30 --output mininet_12.npz
```

The runner provides deterministic 12-, 24-, and 48-node topologies and the six injected fault types evaluated in the manuscript.

## External benchmark

`scripts/prepare_external_benchmark.py` converts a DejaVu-style public benchmark folder
(Li et al., FSE'22, `graph.yml` + `metrics.csv` + `faults.csv`, per the official
[NetManAIOps/DejaVu](https://github.com/NetManAIOps/DejaVu) README) into the same npz
schema used everywhere else in this repository, so the existing `topokpi train` /
`evaluate` and baseline scripts can run on it unmodified:

```bash
pip install pandas pyyaml   # only needed for this optional adapter
PYTHONPATH=src python scripts/prepare_external_benchmark.py --data-dir data/D --inspect
PYTHONPATH=src python scripts/prepare_external_benchmark.py --data-dir data/D --output external_d.npz
topokpi train --data external_d.npz --config configs/external.json --output runs/external
topokpi evaluate --run runs/external --data external_d.npz
PYTHONPATH=src python scripts/run_rca_baselines.py --run runs/external --data external_d.npz
```

`configs/external.json` records the 40%/20%/40% split ratio used for the public benchmark. Run the converter with `--inspect` before conversion to verify the downloaded column names and use the documented command-line overrides when a benchmark release uses different field names. The five event-level split summaries reported in the manuscript are included in `data/reported/manuscript_results.json`.

## Implementation and evaluation scope

- The proposed detector and RCA procedure follow the equations and settings stated in the manuscript.
- The submitted experimental model contains approximately 0.42 million trainable parameters, independent of topology size because its weights are shared across nodes.
- OLS and VAR(1) are implemented directly in NumPy. The remaining interfaces expose the common input, threshold-selection, and evaluation protocol used for the reported comparisons.
- The revision-table driver macro-averages the 12-, 24-, and 48-node results within each seed and then reports the five-seed mean, sample standard deviation, and 95% t confidence interval.
- The Mininet KPI streams are not included in this public code archive. As stated in the manuscript, they are available from the corresponding author on reasonable request.
- The public DejaVu Dataset D is obtained from its cited Zenodo record and is not redistributed in this archive.

## Test

```bash
python -m unittest discover -s tests -v
```

## Code availability

This archive accompanies the manuscript submitted to *Scientific Reports* and enables reviewers and readers to inspect and run the core algorithm: the self-supervised detector in `src/topokpi/model.py`, topology-prior RCA ranking in `src/topokpi/rca.py`, and multi-granularity scoring and explanation procedures in `src/topokpi/metrics.py` and `src/topokpi/explainability.py`.
