# START HERE — step-by-step setup (Ubuntu Linux)

**Read this entire page before running any model.** No Linux or machine-learning experience is assumed. We do not overwrite the original research papers or automatically download private/licensed data.

## 0. What you need and what is possible

Your supplied hardware capture reports an **NVIDIA RTX 4000 Ada Generation (~20 GB GPU memory)**, an NVIDIA driver version **580.173.02**, driver-reported CUDA **13.0**, **31 GB RAM** and **28 logical CPU threads**. This is appropriate for testing the baselines, though larger experiments may need smaller batches. Driver-reported CUDA 13.0 is **not** the installed PyTorch CUDA toolkit version; the scripts use a compatible official PyTorch **2.4.1 / cu121** binary. You do not need to downgrade the NVIDIA driver or install the old CUDA 10.2 toolkit.

Setup is designed for **Ubuntu Linux** (or genuine Linux with equivalents), not native Windows. If you run Ubuntu in WSL2, ensure your GPU is visible before installing, and do not try to run real Mininet fault injection on WSL2 unless networking and Open vSwitch are working.

**Important honesty:** In our review environment, no NVIDIA GPU or download access was available. We verified the TopoKPI CPU pipeline and the COSI input checks, but cannot certify GPU training or PyG/Poetry installation on your specific machine without you executing them.

## 1. Open a terminal in the actual repository

1. Open Ubuntu **Terminal** (shortcut: Ctrl+Alt+T).
2. Find the folder containing `README.md`, `researchPapers/` and `baselineReproduction/`.
3. In the commands below, replace `REPLACE_WITH_YOUR_REPOSITORY_PATH` with **the full path** shown in your Files application.

```bash
cd "REPLACE_WITH_YOUR_REPOSITORY_PATH"
pwd
ls baselineReproduction
```

You should see `topology_kpi_reproduction`, `On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection`, `cloud_rca_reproduction`, and `setup`. If `ls` fails, **stop** and correct your folder path. Keep this terminal at the repository root until the instructions tell you to change folders.

## 2. Install Ubuntu build prerequisites and pyenv (one time)

`pyenv` lets each baseline use Python **3.10.13** without changing the operating system Python. This is especially useful for older GDN, COSI and GraGOD code.

First run:

```bash
sudo apt update
sudo apt install -y git curl build-essential make gcc libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils \
  tk-dev libffi-dev liblzma-dev ca-certificates unzip
```

If `pyenv --version` already works, **skip** the next install and shell configuration commands. Otherwise run:

```bash
git clone https://github.com/pyenv/pyenv.git "$HOME/.pyenv"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$HOME/.bashrc"
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$HOME/.bashrc"
echo 'eval "$(pyenv init - bash)"' >> "$HOME/.bashrc"
```

**Close this terminal and open a new terminal** (this loads `.bashrc`). Return to your repository as in Step 1 and verify:

```bash
pyenv --version
nvidia-smi
```

The `nvidia-smi` command should show your RTX 4000 Ada GPU. If it fails, read `setup/TROUBLESHOOTING.md` before choosing GPU mode.

## 3. Pick ONE baseline first

Run each `setup.sh` from the **repository root**, in a terminal with pyenv available. It may download several gigabytes and take time to compile Python. Setup scripts do not use `sudo` or modify global Python packages. Each creates its own `.venv` to avoid dependency conflicts.

**Recommended order:** begin with TopoKPI (verified CPU run), then COSI (graph/data preparation), then GDN/GraGOD where needed. The third paper cannot start without the authors' files.

```bash
# Paper 1 — install + run bundled unit tests
bash baselineReproduction/topology_kpi_reproduction/setup.sh gpu

# Paper 3 — COSI graph/training code (separate isolated environment)
bash baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/setup.sh gpu

# Paper 3 supporting original GDN baseline
bash baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GDN/setup.sh gpu

# Paper 3 supporting GraGOD training framework (Poetry, Git access needed)
bash baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GraGOD/setup.sh gpu

# Paper 3 COSI-specific nested fork; needed for its Cisco implementation
bash baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/GraGOD/setup.sh gpu
```

**Do not run all four in one go.** First finish `01_TOPOKPI.md`, then read `02_COSI.md`. After each setup, run the `source .../.venv/bin/activate` command given in that guide. A script can use the `cpu` argument instead of `gpu` for a CPU-only installation:

```bash
bash baselineReproduction/topology_kpi_reproduction/setup.sh cpu
```

**What success looks like:** `SUCCESS` after dependency checks or tests; if `ERROR`, do not assume the setup succeeded. The installed environments are intentionally excluded from Git and `output.zip`.

## 4. Keep your results scientifically trustworthy

- **Unit tests** prove individual functions behave as coded; they do not prove paper reproduction.
- **Synthetic demonstration** proves the TopoKPI training pipeline runs. It does not match the Mininet or Train-Ticket experiments.
- **MSL/Pearson smoke test** checks that the COSI/GDN training wiring accepts a graph. MSL/Pearson is NOT the COSI graph-construction method.
- **Paper reproduction** requires the same dataset, splits, pre-processing, graph construction, metrics, seeds and model hyperparameters as the paper. Record any necessary deviations.
- **Training isolation:** fit normalization, graph statistics, threshold tuning, and model parameters only on the appropriate training/validation partitions. Do not optimize based on test labels.

See `GIT_WORKFLOW.md` for exact Git commits, `TROUBLESHOOTING.md` for errors, `01_TOPOKPI.md`, `02_COSI.md`, `03_CLOUD_RCA.md` for each baseline.
