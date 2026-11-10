# SKILLS.md

## Purpose

This file records the technical workflow that repeatedly proved effective while building and debugging `Network Anomaly Detection & Root Cause Localisation`.

It is intended to guide future AI agents through the same engineering process rather than merely describing the architecture.

You are given the two part literature review PDFs. 

Phase1 PDF file contains a preliminary overview of the literature review. It contains the information for our 3 base papers and the gap that can be extracted from a combination of all three papers. Phase2 PDF is an indepth view and a proper literature review for our research project. It also contains information on on the contribution we want to be able to make.

Each paper has it's own dataset. It's own source code. I have checked all 3 papers. The dataset for 2 of them is atleast given with Github links for the repository. My memory is hazy but for 1 paper which is probably the '.... KPI anomaly' paper, the dataset isn't available.
- In the case of any dataset not available publicly, check if there is a code that can help recreate the dataset or download the dataset.
- In the case that source code isn't available publicly but the paper describes exact procedure? Well then recreate the methodology and work as closely as possible. Although doing this would create the a few problems and let me instruct you on how to counter them as our aim is to recreate the work of the authors as closely as possible. So here you go:
1. If the exact code library isn't disclosed, use what most accurately resembles.
2. If evaluation metrics are not given then use 5 of the most appropriate metrics for the task. 
3. If exact model files aren't given we must recreatethem by running the training pipeline ourselves.
4. Atleast 1 paper's authors have informed us on the specifics of the machine they used but we are limited to our own machines so our work needs to optimise thecode to be able to function on our own machine. Like if they used a 64 GB VRAM GPU then there batch size must have been larger although on our machines with 20/10 GB VRAM, our batch size will be different.
5. If the authors have not informed us of there exact training configurations then we must choose the most appropriate ones.
6. If the authors have not disclosed the hyper parameters, then we will perform hyper parameter evaluation using either bayesian or `optuna` although `optuna` is preferred. We will be choosing the best hyper parameters of the best model from our hyper parameter optimisation.

---

# Skill 1 — Establish Repository Ground Truth

## Objective

Understand the real state of the repository before changing it.

## Procedure

### Step 1 — Resolve the repository root

Do not assume a path.

Historically the project has been used from environments including:

```text
~/Network-Anomally-Detection_Root-Cause-Localization_Generalisation
```

and Windows/WSL locations.

Resolve the active repository root first.

### Step 2 — Inspect Git state

Check:

```bash
git status
```
Github URL: [https://github.com/ib-hussain/Network-Anomally-Detection_Root-Cause-Localization_Generalisation](https://github.com/ib-hussain/Network-Anomally-Detection_Root-Cause-Localization_Generalisation)

Then inspect relevant branches, recent commits, and changed files.

Before destructive Git operations determine whether valuable local-only files exist.

### Step 3 — Inspect repository structure

Identify:

```text
src/code/
src/data/
outputs/
baselineReproduction/
setup/
```


### Step 4 — Read the implementation and its consumers

For the target module inspect:

```text
implementation
configuration
CLI
input loader
artifact loader
tests
documentation
```

Do not modify a model file in isolation when another module relies on its exact schema.

---

# Skill 2 — Verify the Runtime

Before diagnosing model logic, verify the environment.

Check:

```text
Python version
virtual environment
PyTorch version
CUDA availability
GPU identity
available VRAM
CPU
RAM
disk space
working directory
```

The project has historically used Python `3.12.7`.

The major remote GPU environment used an NVIDIA RTX 3090 Ti with approximately 24 GB VRAM.

Do not assume that environment is always active.

Code must detect the current runtime.

---

# Skill 3 — Compile Before Running

For each changed Python module:

```bash
python -m py_compile path/to/file.py
```

Compilation is only the first gate.

Passing compilation means:

```text
syntax is valid
```

not:

```text
the algorithm is correct
```

Continue with runtime verification.

---

# Skill 4 — Inspect Data Before Training

Before launching a model:

1. verify input paths;
2. inspect column names;
3. inspect dtypes;
4. inspect timestamps;
5. inspect missing values;
6. inspect duplicate rows;
7. inspect finite values;
8. inspect row counts.


Never infer alignment simply because two files have similar row counts.

---

# Skill 5 — Build a Small Real-Data Smoke Path

Before HPO or full training, run a small representative subset.

The smoke test should exercise the real path:

```text
loader
→ preprocessing
→ model forward
→ loss / prediction
→ output generation
→ serialization
```

A useful smoke test verifies more than model construction.

It should catch:

* missing files;
* shape mismatches;
* alignment failures;
* device errors;
* dtype errors;
* output-writing errors.

Prefer a small real dataset over fabricated financial data.

---

# Skill 6 — HPO Before Full Training

For trainable models, use the sequence:

```text
smoke
→ small HPO validation
→ persistent HPO
→ inspect trials
→ select best parameters
→ full training
```

Use Optuna TPE where the module follows the established project pattern.

Persist studies in SQLite.

A rerun should resume the study unless explicitly started fresh.

Provide a deliberate fresh-study option rather than deleting databases manually.

---

# Skill 7 — Guard Every HPO Objective

Inside the objective function validate all major numerical outputs.

Reject:

```text
NaN
+Inf
-Inf
empty predictions
invalid losses
invalid metric values
```

before reporting results to Optuna.

Do not let one corrupt training state poison a study.

When possible, report intermediate metrics and support pruning.

---

# Skill 8 — Train Resumably

Full training should preserve enough state to continue.

At the end of each epoch save:

```text
epoch
model state
optimizer state
scheduler state
best metric
training configuration
```

Maintain a discoverable latest checkpoint.

Resume logic should:

1. detect an existing checkpoint;
2. load state;
3. restore epoch;
4. restore optimisation state;
5. continue with the next epoch.

Never label a restart as a resume.

---

# Skill 9 — Export Inference Artifacts Explicitly

Distinguish:

```text
training checkpoint
trained model export
frozen model
embedding cache
prediction cache
scaler/PCA state
manifest
```

Where required preserve both frozen and unfrozen exports.

A deployment path must consume the intended artifact type rather than whichever file happens to load successfully.

---

# Skill 10 — Validate Embedding Artifacts

For each generated embedding array inspect:

```python
shape
dtype
np.isfinite(...).all()
```

Verify metadata:

```text
row count
timestamps
identifiers
split
```

For very large artifacts, generate a manifest rather than repeatedly loading everything manually.

---

# Skill 11 — Debug Numerically

When model results are suspicious, print or log the first meaningful numerical evidence.

Examples:

```text
input shape
min/max
mean/std
finite ratio
label distribution
prediction distribution
loss
class probabilities
join counts
unmatched IDs
```

Do not infer numerical correctness from absence of exceptions.

---

# Skill 12 — Verify the UI Against Raw Output

For representative cases compare:

```text
raw inference JSON
vs
displayed UI values
```

The UI must not silently round, transform, or reinterpret a value into a different conclusion.

---

# Skill 13 — Safe Git Workflow

Before committing:

```text
git status
```

Confirm that only intended files changed.

For generated data/model files determine whether they should be:

```text
ignored
tracked normally
```

Do not commit `.env`.

Before destructive cleanup inspect ignored files separately.

---

# Skill 14 — Recover from Git Mistakes

When working-tree state becomes unsafe:

1. stop issuing destructive commands;
2. inspect branches;
3. inspect stashes;
4. inspect reflog;
5. identify Git-tracked recoverable content;
6. identify LFS content;
7. identify untracked/ignored content;
8. recover to a separate safe location first;
9. verify recovered artifacts;
10. only then rebuild the working tree.

Do not perform cleanup while recovery is still underway.

---

# Skill 15 — Long Remote Jobs

For remote training, embedding generation, or large Git/LFS transfers use persistent terminal sessions.

Typical pattern:

```text
SSH
→ tmux
→ activate environment
→ start job
→ detach
```

A network disconnect should not terminate a multi-hour computation.

---

# Skill 30 — Documentation After Implementation

Once a component is stable, document:

```text
purpose
inputs
outputs
data source
split scheme
algorithm/model
HPO
training
checkpoint paths
artifact formats
inference interface
validation
known limitations
```

Documentation should represent the current architecture.

Move obsolete design decisions into a clearly marked historical section rather than mixing them with active instructions.

---

# Final Workflow

The preferred end-to-end problem-solving loop is:

```text
1. Inspect repository
2. Inspect runtime
3. Inspect data
4. Trace full dependency path
5. Identify root cause / required change
6. Make minimal modification
7. Compile
8. Run targeted tests
9. Run real-data smoke test
10. Verify restart/resume behaviour
11. Run HPO where applicable
12. Run full computation
13. Validate artifacts numerically
14. Validate multimodal alignment
15. Validate inference
16. Inspect Git diff
17. Update documentation
18. Commit only understood changes
```

The governing principle is:

> **Never spend hours recomputing something that could have been validated in minutes, and never destroy state that could have saved hours of recomputation.**
