# AGENTS.md

## Project Identity

**Project:** `Network Anomaly Detection & Root Cause Localisation`
**Owner / primary developers:** Ibrahim Hussain, Sameer Khan
**Domain:** Data Science, Computer Networks, multimodal machine learning, advanced deep learning.

This file defines how AI coding agents must work inside this repository.

The goal is not merely to produce code that runs. The repository must remain:

* reproducible;
* leakage-safe;
* resumable;
* inspectable;
* modular;
* explainable;
* compatible with both CPU and CUDA where practical;
* safe to operate on expensive datasets and model artifacts;
* recoverable after interruptions;
* understandable by another engineer without relying on chat history.

---

# Core Engineering Philosophy

## 1.1 Evidence before edits

Never begin by rewriting a module from assumptions.

Before modifying code:

1. inspect the repository tree;
2. inspect the relevant implementation;
3. inspect its configuration;
4. inspect the data schema or saved artifact it consumes;
5. inspect existing tests;
6. inspect logs/checkpoints from previous runs;
7. determine whether the requested behaviour already partially exists.

Prefer understanding and extending the current system over replacing it.

---

## 1.2 Preserve working state

The repository contains expensive-to-reproduce artifacts.

Never casually delete:

* datasets;
* processed datasets;
* checkpoints;
* embeddings;
* Optuna studies;
* model exports;
* cached predictions;
* manifests;
* `.env`;
* local credentials;
* logs needed for recovery.


Ignored files may be the most expensive files in the project.

---

## 1.3 Root-cause debugging

Do not patch symptoms until the complete failing path is understood.

Trace:

```text
input
  ↓
loader
  ↓
preprocessing
  ↓
model/module
  ↓
serialization/cache
  ↓
final output
```

For shape, dtype, path, NaN, checkpoint, alignment, or schema errors, identify the first point where reality diverges from the expected invariant.

Do not insert arbitrary conversions merely to silence exceptions.

---

## 1.4 Minimal changes first

Prefer the smallest correct modification that preserves:

* public interfaces;
* checkpoint compatibility;
* output schemas;
* existing configuration;
* prior results.

Large refactors require evidence that the current structure itself is causing the problem.

---

# Project Architecture

The project architecture is not yet final but it is being worked upon and ou will be updated with the projectarchitecture once it is finalised and we can start work on it.

---

# HPO-First Training

Expensive model training should follow:

```text
inspect
→ smoke
→ HPO
→ select parameters
→ full training
→ validation
→ export
```

Do not jump directly to arbitrary full-training hyperparameters when the component supports HPO.

Preferred HPO stack:

* Optuna;
* TPE sampler;
* persistent SQLite studies.

Studies must be resumable.

Provide a deliberate `--fresh` mechanism where appropriate rather than overwriting an existing study by accident.

Only one major GPU HPO workload should normally run at once.

---

# Failed-Trial Discipline

A bad trial should fail early.

Reject or prune trials when encountering:

* NaN;
* Inf;
* non-finite loss;
* invalid tensor dimensions;
* empty datasets;
* unusable predictions;
* impossible class distributions;
* missing required artifacts.

Do not feed non-finite metrics into Optuna.

---

# Resumability

Every expensive stage should be restartable.

Training should normally provide:

* checkpoint after each epoch;
* a stable latest-checkpoint location;
* optimizer state;
* scheduler state where applicable;
* epoch number;
* model state;
* enough metadata to continue correctly.

A resumed run must continue rather than silently restart.

For multi-stage pipelines, persist stage-level state where practical.

---

# Model Exports

Where the component design requires it, preserve both:

```text
epoch_{trained/validated}_n.pt
```

or the repository's established equivalent.

Do not confuse:

* a training checkpoint;
* a frozen inference model;
* cached predictions;
* replay data.

They serve different purposes.



---

# UI Principles

Transparency is the default.

The interface should support:

* final recommendation;
* confidence;
* position sizing;
* system-level explanation;
* per-module outputs;
* selectable module details;
* raw structured/JSON output.

Do not hide intermediate reasoning behind a single polished narrative.

---

# CLI Design

Important modules should be usable both:

```text
as imports
```

and

```text
as executable CLIs
```

Where appropriate, support operations conceptually equivalent to:

```text
inspect
smoke
hpo
train
predict
validate
```

Use the repository's existing CLI style rather than inventing incompatible conventions.

Useful recurring options include:

```text
--repo-root
--fresh
```

when appropriate.

Do not hardcode one developer's absolute path.

---

# CPU + CUDA Behaviour

Code should work on CUDA where acceleration is useful but should not become unnecessarily unusable on CPU.

Always prefer CUDA if a PC has it instead of using CPU so that even a smoke-test runs on CUDA and saves time.

Device handling must be explicit.

Typical pattern:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

but follow existing abstractions where present.

CPU work should be parallelised sensibly for:

* preprocessing;
* data preparation;
* statistical risk computation;
* manifest generation;
* validation.

Do not move trivial work to GPU merely because CUDA is available.

---

# Real Data Only

Tests may use tiny subsets of real project data.

Do not create fake financial observations to make a production pipeline appear functional unless a unit test explicitly requires synthetic fixtures.

For integration/smoke testing, prefer:

```text
small real subset
```

over:

```text
invented representative data
```

---

# Validation Before Expensive Execution

Recommended sequence after code changes:

```text
syntax / import validation
→ focused unit test
→ small real-data smoke test
→ resume-path test
→ HPO smoke
→ full operation
```

At minimum, compile changed Python files:

```bash
python -m py_compile <file>
```

Then execute the smallest meaningful path that exercises the modification.

---

# Artifact Validation

Never trust file existence alone.

For generated embeddings, predictions, or models validate:

* path;
* row count;
* shape;
* dtype;
* finite values;
* metadata alignment.

For large outputs, generate or preserve manifests.

---

# Shell Interaction Style

Ibrahim works in both WSL, Linux shells and Windows PowerShell.
Sameer works in both WSL and Windows PowerShell.

Commands must match the active shell.

Do not provide Bash syntax to PowerShell or PowerShell syntax to Bash.
Bash commands must be preferred. Bash wokring and working with Bash is preferred.
Commands should generally be:

* directly copyable;
* concise;
* explicit;
* single-line when practical.

Avoid Bash commands split using trailing `\` unless there is a compelling reason.

For long-running remote Linux operations, use persistent terminal sessions such as `tmux`.

---

# Git Safety

Try your absolute best to keep committing small changes every time something new happens. You are allowed to make a `.gitattributes` file but you are not allowed to declare any file for Git LFS. Data will stay local and any non-LFS files will stay local, `.json` files can be committed as well as the files such as `status.json`. No file above the Git normal file size should be committed. You are forbidden from using Git LFS. Declare every file's proper handling if it's committed in the `.gitattributes` file. Use the `.gitignore` file to ignore files that are not needed to be committed. Use the `.gitattributes` file to declare the proper handling of files that are committed.

Preserve local-only artifacts.

`.env` must stay local and uncommitted.

Before large LFS pushes, inspect what is actually going to be transferred.

Always check `git status` before making commits.

---

# Repository Hygiene

Expected File Structure:
```bash
ibrahim@IbLaptop{/mnt/d/Downloads/Repositories/Network-Anomally-Detection_Root-Cause-Localization_Generalisation}: tree -L 4 .
.
├── app.py                      (The final Streamlit app that will be used to demonstrate the work in a web app)
├── baselineReproduction/       (Each folder in this is for reproducing the work of each paper)
│   ├── cloud_rca_reproduction/
│   │   └── setup.sh
│   ├── On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/
│   │   ├── 5G3E-dataset/
│   │   │   ├── README.md
│   │   │   ├── version1/
│   │   │   └── version2/
│   │   ├── COSI/
│   │   ├── GDN/
│   │   └── GraGOD/
│   ├── README.md
│   ├── setup/
│   │   ├── 01_TOPOKPI.md
│   │   ├── 02_COSI.md
│   │   ├── 03_CLOUD_RCA.md
│   │   ├── CHANGELOG.md
│   │   ├── common.sh
│   │   ├── GIT_WORKFLOW.md
│   │   ├── README.md
│   │   ├── TROUBLESHOOTING.md
│   │   └── VALIDATION_REPORT.md
│   ├── topology_kpi_reproduction/  (This folder is for reproducing the work of the paper `Topology-aware self-supervised multivariate KPI anomaly detection and root cause node localization for network operations`)
│   │   ├── ARCHIVE_MANIFEST.md
│   │   ├── CITATION.cff
│   │   ├── configs/
│   │   ├── data/
│   │   │   ├── processed/
│   │   │   ├── raw/
│   │   │   └── reported/
│   │   ├── docs/
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── scripts/
│   │   ├── setup.sh
│   │   ├── src/
│   │   └── tests/
│   └── topology_kpi_reproduction_zenodo_v1.0.0.zip
├── outputs/                    (Each folder is for the results of the operations of the work of our own and from reproduction of the work of the papers)
│   ├── figs/                   (This folder is for storing any possible figures and visualizations)
│   ├── embeddings/             (This folder is for storing any possible model embeddings that we might want to use or that we obtain)
│   │   ├── reproduction/       (This folder is for storing the embeddings from our reproduction of the papers)
│   ├── hpo/                    (This folder is for storing the results of our hyper parameter optimisation)
│   │   ├── reproduction/       (This folder is for storing the results of our hyper parameter optimisation from our reproduction of the papers)
│   │   │   ├── trial_1/
│   │   │   ├── trial_2/
│   │   │   └── trial_n/
│   │   ├── README.md
│   │   ├── trial_1/            (Each folder is for storing the results of the trials of our hyper parameter optimisation)
│   │   ├── trial_2/
│   │   └── trial_n/
│   ├── logs/                   (This folder is for storing the logs of our work and from reproduction of the papers)
│   │   ├── reproduction/       (This folder is for storing the logs from our reproduction of the papers)
│   │   ├── data/               (This folder will store our logs from data downloading to processing operations)
│   │   ├── inference/          (This folder is for storing logs of our inference operations)
│   │   ├── README.md
│   │   ├── setup/              (This folder is for storing logs from our setup of environment when we first start our work)
│   │   ├── test/               (This folder is for storing logs of the testing of our models)
│   │   ├── train/              (This folder is for storing logs of the training of our models)
│   │   └── validate/           (This folder is for storing logs of the validation of our models)
│   ├── model/                  (This folder will store our model files. Each epoch of our training will be stored here. Those epochs we will use for performing validation and testing of our models. We can also store the best model file here for inference operations. We can also store post-validation epoch files here as well)
│   │   ├── reproduction/       (This folder is for storing the model files from our reproduction of the papers)
│   │   │   └── epoch_01.pt
│   │   ├── epoch_01.pt.replaceThis
│   │   └── README.md
│   └── README.md
├── README.md
├── researchPapers/             (This folder is for storing the research papers and documentation of our entire project. This includes our base papers, our Literature review, supporting documents and any explainations for our methodology in the form of markdown documents which will be accessible via the main README.md file in root of the repository for full-flegded understanding of the public)
│   ├── On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection.pdf
│   ├── Phase_1_Literature_Review.pdf
│   ├── Phase2_GNN_AIOps_Research_Methodology.docx
│   ├── Phase2_GNN_AIOps_Research_Methodology.pdf
│   ├── Root cause analysis of cloud platform faults based on anomaly correlation.pdf
│   └── Topology-aware self-supervised multivariate KPI anomaly detection and root cause node localization for network operations.pdf
├── setup/                      (This folder is for storing the setup scripts and documentation for our work)
└── configs/                    (Our project configurations in the form of YAML files. This includes our hyper parameter optimisation configurations, training configurations and any other configurations we might need for our work)
│   ├── HPO_config.yaml
│   ├── train_config.yaml
│   ├── train_10gb.yaml         (Our training configuration for a machine with 10 GB VRAM GPU(~9 GB usable))
│   ├── train_20gb.yaml         (Our training configuration for a machine with 20 GB VRAM GPU(~19 GB usable))
│   └── README.md
└── src/                        (The source code for our work. This includes the code for our training, testing, validation and inference operations. It also includes the code for our data processing operations)
    ├── code/                   (This folder is for storing the code for our work. This includes the code for our training, testing, validation and inference operations)
    │   ├── gnn/                (This folder contains the entire code for GNNs model in PyTorch. This is is written in form of header files)
    │   │   └── README.md
    │   ├── inference.py
    │   ├── README.md
    │   ├── test.py
    │   ├── train.py
    │   ├── streamlit_helper.py
    │   └── validate.py
    └── data/                   (This folder is for storing the code for our data processing operations. This includes the code for our data downloading, data processing and data augmentation operations)
        ├── processed/          (This folder is for storing the processed data from our processing operations. This includes the data that has been processed and is ready for our training, validation, and testing operations)
        │   └── README.md
        ├── processing1.py
        ├── processing2.py
        ├── raw/                (This folder is for storing the data from our downloading operations. This includes the data from the internet and any other sources)
        │   └── README.md
        └── README.md

```

---

# Documentation Style

Use clear technical English.

Prefer:

* explicit architecture;
* data shapes;
* paths;
* commands;
* inputs;
* outputs;
* failure modes;
* reproducibility instructions.

Avoid vague statements such as:

> The data is then processed.

Instead state what transformation occurs, where its fitted state comes from, and what output is produced.

British English is preferred for project documentation.

---

# Debugging Response Style

When reporting a problem:

1. state what actually failed;
2. distinguish root cause from symptoms;
3. cite the evidence;
4. explain what must change;
5. make the smallest safe modification;
6. show how to verify it;
7. do not declare success until verification succeeds.

Do not call a pipeline "fixed" because compilation passes.

---

# Definition of Done

A change is complete only when the relevant subset of the following has been demonstrated:

* syntax/import passes;
* targeted test passes;
* real-data smoke path passes;
* checkpoint/resume path works if relevant;
* CPU path remains valid where required;
* CUDA path remains valid where required;
* output schema is preserved or deliberately versioned;
* outputs contain no unexpected NaN/Inf;
* manifests/alignment checks pass;
* documentation is updated;
* no unrelated files were modified;
* Git diff is understood;
* expensive user data was not destroyed.

---

# Agent Behaviour Summary

When working in this repository:

**Inspect first.
Preserve state.
Trace the entire path.
Use real evidence.
Fit only on training data.
Smoke-test before full runs.
HPO before arbitrary training.
Checkpoint expensive work.
Expose intermediate outputs.
Keep the narrator separate from the decision system.
Protect ignored data.
Do not claim success without verification.**

# Rules
Preferred artifact forms:

```text
*.csv
*.json
*.jsonl
```

unless an existing module explicitly requires something else.

When asking user for `.zip` of the repository, explicitly give user the bash command for making the zip and then ask user to upload that file. The command you give should include names of all files which you want. 

For temporary fixes or patches, write a file with the code and give user filename, and code and execution instrcutions. Such as:
'''
To fix this problem make this file `fix.sh` with the following code:
```bash
....
```

Then run the following command to execute the fix:
```bash
./fix.sh
```

'''

In this way ou must help user to fix the problem and then run the fix. You must not just give user the code and ask user to run it. You must give user the code, filename and execution instructions.