# Paper 1 — Luo et al., Topology-aware KPI anomaly detection + root-cause node localization

**Status:** The submitted code is present under `baselineReproduction/topology_kpi_reproduction/`; its **original Mininet KPI dataset is NOT included**. Tests and an end-to-end synthetic run worked on CPU in the audit environment. Published five-seed F1/A@1 values are not automatically reproduced by running synthetic data.

## A. Install from the repository root

Make sure `pyenv --version` works. If not, complete `setup/README.md` Steps 1–2.

```bash
bash baselineReproduction/topology_kpi_reproduction/setup.sh gpu
```

The installer creates `baselineReproduction/topology_kpi_reproduction/.venv`, installs torch and package `topokpi`, then runs all bundled unit tests. You should see `Ran 12 tests` and `OK`. For CPU-only setup, replace `gpu` with `cpu`.

## B. Activate this baseline's environment

The following commands assume you start at the repository root:

```bash
cd baselineReproduction/topology_kpi_reproduction
source .venv/bin/activate
python --version
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

`python --version` should print 3.10.13. For GPU, `CUDA: True` should appear. If False, check `nvidia-smi` and the troubleshooting guide. Keep this terminal open; the `( .venv )` prompt indicates the correct environment.

## C. First working end-to-end trial: synthetic only

**Run every command below while inside `topology_kpi_reproduction/`:**

```bash
mkdir -p runs
python -m topokpi.cli generate-demo --output runs/demo.npz --nodes 12 --steps 320 --seed 17
python -m topokpi.cli validate-data --data runs/demo.npz
python -m topokpi.cli train --data runs/demo.npz --config configs/smoke_test.json --output runs/demo_run --device cpu
python -m topokpi.cli evaluate --run runs/demo_run --data runs/demo.npz --device cpu
```

This is the CPU sequence verified during the audit. A 320-step demo is sufficient; **the generator requires at least 300 steps**. The smoke config uses **two epochs**, not the research settings. You should see JSON with a detection block (precision, recall, F1) and RCA block (A@1, A@3, A@5). These numbers measure a tiny synthetic fixture and must never be presented as published reproduction results.

To exercise the GPU only after the CPU test passes, repeat the last two commands with a different output directory and `--device cuda`:

```bash
python -m topokpi.cli train --data runs/demo.npz --config configs/smoke_test.json --output runs/demo_gpu --device cuda
python -m topokpi.cli evaluate --run runs/demo_gpu --data runs/demo.npz --device cuda
```

If you receive `CUDA not available` or out-of-memory errors, follow `TROUBLESHOOTING.md`. Never overwrite a CPU test run with a GPU test run.

## D. Next stage: author-equivalent data, not included

1. Read the paper in `researchPapers/Topology-aware self-supervised multivariate KPI anomaly detection and root cause node localization for network operations.pdf`. Verify **12/24/48 monitored nodes, 8 KPIs, a 30-second sample interval, 20-step windows, and six fault types**.
2. **For exact published results**, request the actual Mininet KPI streams, event labels, root-cause node labels, topology and train/validation/test split files from the corresponding author via their official article. A raw PDF or `data/reported/*.csv` cannot reconstruct the exact streams.
3. For a *new controlled data collection* (not guaranteed identical to the authors' campaign), install the network emulator on an isolated Linux machine and review what the command would do:

```bash
sudo apt install -y mininet openvswitch-switch iperf3 iputils-ping
PYTHONPATH=src python scripts/mininet_campaign.py --nodes 12 --steps 1200 --interval 30 --output runs/mininet12.npz --dry-run
```

4. **Only when you understand the experiment and are using a Linux host where Mininet and Open vSwitch work**, run the real collection. The command needs elevated privileges to create virtual network interfaces:

```bash
sudo env PYTHONPATH="$PWD/src" "$PWD/.venv/bin/python" scripts/mininet_campaign.py \
  --nodes 12 --steps 1200 --interval 30 --output "$PWD/runs/mininet12.npz"
```

5. Validate any dataset produced or received:

```bash
python -m topokpi.cli validate-data --data runs/mininet12.npz
```

6. Check that it contains `x`, `labels`, `root_causes`, `fault_types`, `a_enc`, `w_rca`, `timestamps`; check KPI order and counts. If successful, train on it in a *separate output folder*:

```bash
python -m topokpi.cli train --data runs/mininet12.npz --config configs/paper.json --output runs/mininet12_seed17 --seed 17 --device cuda
python -m topokpi.cli evaluate --run runs/mininet12_seed17 --data runs/mininet12.npz --device cuda
```

Data collected yourself must be labeled **independent replication**, not **exact original dataset reproduction**. Compare against article tables only after verifying their protocols.

## E. Public Train-Ticket / DejaVu external benchmark

The TopoKPI repository documents an adapter to the DejaVu Dataset D layout (`graph.yml`, `metrics.csv`, `faults.csv`). Find the official dataset linked from https://github.com/NetManAIOps/DejaVu and download it according to its own access terms. It is **not** included in the ZIP.

After downloading and extracting the three files under `data/D/`:

```bash
python -m pip install pandas pyyaml
PYTHONPATH=src python scripts/prepare_external_benchmark.py --data-dir data/D --inspect
PYTHONPATH=src python scripts/prepare_external_benchmark.py --data-dir data/D --output runs/external_d.npz
python -m topokpi.cli validate-data --data runs/external_d.npz
python -m topokpi.cli train --data runs/external_d.npz --config configs/external.json --output runs/external_seed17 --device cuda
python -m topokpi.cli evaluate --run runs/external_seed17 --data runs/external_d.npz --device cuda
```

`--inspect` is essential: check the source column mapping and node labels before conversion. If the release's fields differ, use the converter's `--help` overrides rather than guessing.

## F. Paper-level checks after data is available

Read `topology_kpi_reproduction/README.md` for the existing five-seed, contamination, ablation, RCA baselines, statistics, comparison and explanation scripts. Re-run each experiment with the authors' input data. Use explicit output directory names and preserve run metadata. To exit this Python environment, run `deactivate`.
