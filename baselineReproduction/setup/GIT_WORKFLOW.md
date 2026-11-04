# Git workflow — apply only changed files and commit in small, reversible steps

These exact commands are for the **repository root** (the folder containing `.gitignore` and `baselineReproduction/`). They preserve your research history. They do not automatically commit or delete anything. One `output.zip` contains only added/modified files.

## Part 1 — Protect your existing work before extracting

Open Ubuntu Terminal, change to the repository root (see `setup/README.md` Step 1), and run:

```bash
git status --short
git branch --show-current
```

If there are unrelated uncommitted changes, **review** them; either commit your own work separately or make a safe backup. Do not use `git reset --hard` or `git clean -fd`. Make an integration branch:

```bash
git switch -c chore/baseline-setup-alignment
```

If the branch already exists, use a different descriptive name. If this repository does not have a `.git/` history yet, run `git init` and `git add` / `git commit` your original files first (after checking that private data and oversized files are not staged).

## Part 2 — Extract the patch ZIP

Put `output.zip` somewhere easy to locate, e.g. `~/Downloads/output.zip`. From the **repository root**:

```bash
unzip -l "$HOME/Downloads/output.zip"
unzip -o "$HOME/Downloads/output.zip" -d .
git status --short
```

The ZIP uses **paths relative to your repository root**; do not extract it inside `baselineReproduction/`. The archive deliberately does not contain unchanged paper PDFs, code, datasets or `.venv` folders.

## Part 3 — Commit changes in checkpoints

Use **one command group at a time**, run `git diff --cached --stat` before each commit if you want to review the staging. You can copy and paste these commands as written, from your repository root. Each `git add` lists exact patch files only.

**Commit 1 — setup helper and entrypoint documentation:**

```bash
git add .gitignore README.md baselineReproduction/README.md \
  baselineReproduction/setup/README.md baselineReproduction/setup/common.sh
git diff --cached --stat
git commit -m "docs(baselines): add setup overview and isolated Python environment helper"
```

**Commit 2 — Paper 1 reproducible CPU workflow:**

```bash
git add baselineReproduction/topology_kpi_reproduction/setup.sh \
  baselineReproduction/setup/01_TOPOKPI.md
git diff --cached --stat
git commit -m "build(topokpi): add verified baseline setup and demo runbook"
```

**Commit 3 — COSI CLI and input repair:**

```bash
git add baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/Start.py \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/data_io.py \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/models/GDN.py \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/short.ipynb \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/prepare_smoke_data.py \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/tests/test_data_io.py \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/setup.sh
git diff --cached --stat
git commit -m "fix(cosi): validate dataset inputs and preserve supplied graph execution"
```

**Commit 4 — GDN and GraGOD compatibility:**

```bash
git add baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GDN/setup.sh \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GDN/install.sh \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GDN/run.sh \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GraGOD/setup.sh \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GraGOD/train.sh \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GraGOD/predict.sh \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GraGOD/research/datasets/20250217_metrics_analysis.py \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/GraGOD/setup.sh \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/GraGOD/models/train.py \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/GraGOD/datasets/config.py \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/GraGOD/datasets/graph.py \
  baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/GraGOD/tests/test_graph_input.py
git diff --cached --stat
git commit -m "build(graph-baselines): add modern GDN and GraGOD setup scripts"
```

**Commit 5 — Missing paper and final research runbooks:**

```bash
git add baselineReproduction/cloud_rca_reproduction/setup.sh \
  baselineReproduction/setup/02_COSI.md \
  baselineReproduction/setup/03_CLOUD_RCA.md \
  baselineReproduction/setup/TROUBLESHOOTING.md \
  baselineReproduction/setup/CHANGELOG.md \
  baselineReproduction/setup/GIT_WORKFLOW.md \
  baselineReproduction/setup/VALIDATION_REPORT.md
git diff --cached --stat
git commit -m "docs(reproduction): record data blockers, troubleshooting and rollback procedure"
```

Then verify:

```bash
git status --short
git log --oneline -5
```

A clean `git status` is expected unless you have other pre-existing changes or generated files. If a new output file appears, inspect it and add it to `.gitignore` only if it is genuinely generated data; do not accidentally commit licensed datasets.

## Part 4 — Roll back only one checkpoint if needed

To undo the **latest** already committed patch while keeping an explicit history:

```bash
git revert HEAD
```

Use `git log --oneline` to find an older commit, then `git revert <that-commit-hash>`. Reverting may require manual conflict resolution if later patches edited the same files. Do not delete the repository or use a destructive reset. If a setup script has already installed `.venv` locally, Git revert does not remove it; it is an untracked environment that you may delete **manually** when no longer needed:

```bash
# OPTIONAL CLEANUP ONLY IF YOU WANT TO REMOVE A LOCAL ENVIRONMENT:
# rm -rf baselineReproduction/topology_kpi_reproduction/.venv
```

**Files that must be deleted due to this patch: NONE.** No removal commands are necessary. Keep the original research files as evidence.
