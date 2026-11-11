# Step 23 — Equation (5) conditional-probability edge weighting

**Paper:** Zhou et al. (2026), Section 4.2, Equation (5), Table 1 and Algorithm 1 (lines 11–17).  
**Scope:** Implement only the *post-causal-graph* edge weighting step. This does **not** implement TTHP, physical-topology-aware RCA, GAT, or LightGBM.

## A. Why this step can proceed now

The public Wireless package has alarm timestamps and alarm types, and your exploratory THP runs created an event-type directed graph (`outputs/thp_18_no_topology_iter5/causal_graph.npy`). Eq. (5) can attach weights to the **existing edges** of that graph. **No ground-truth `DAG.npy` is passed to this script.** The learned THP graph was trained using all events, however, so this is still an **exploratory integration check** and must not be treated as a leakage-free held-out evaluation.

The paper requires slicing alarm data by time and marking whether each alarm type appeared during each time slice. It assigns an edge `i -> j` the conditional fraction:

`weight(i,j) = (# time slices containing both i and j) / (# time slices containing i)`.

Only a causal edge already inferred by TTHP should get that weight. Co-occurrence cannot create or establish causal direction by itself.

**Unspecified author detail:** The paper does not provide an unambiguous time-slice width for the Wireless release. We use 384 equal-width slices within the first 70% of the observed **timestamp range**. That is an explicit exploratory choice, **not** the paper's reported random 7:3 split protocol. Timestamps' units remain unknown. Changing the bin count changes the weights.

## B. Install patch and activate WSL environment

From any folder inside the repo, check that you are in WSL and the environment is active:

```bash
cd "$(git rev-parse --show-toplevel)/baselineReproduction/cloud_rca_reproduction"
source .venv-thp/bin/activate
python --version
```

No new packages are necessary (NumPy and pandas are already installed).

## C. Run the tests first

```bash
python -m unittest discover -s tests -v
```

Your previous 13 tests plus four new tests should make **17 passing tests**, assuming your prior files are unchanged.

## D. Produce exploratory ACG weights

Check that your **learned** 5-iteration graph exists:

```bash
ls -lh outputs/thp_18_no_topology_iter5/causal_graph.npy
```

Then run the command **from `cloud_rca_reproduction/`**:

```bash
python -u -m scripts.build_acg_weights \
  --events data/processed/wireless_thp_events.csv \
  --graph outputs/thp_18_no_topology_iter5/causal_graph.npy \
  --bins 384 \
  --train-fraction 0.7 \
  --output outputs/acg_eq5_exploratory
```

The command creates three outputs:

- `outputs/acg_eq5_exploratory/acg_weights.npy`: 18×18 matrix of weights in [0,1]; zero outside supplied graph edges.
- `outputs/acg_eq5_exploratory/edges.csv`: edge list with source event, target event, conditional weight and source-support count.
- `outputs/acg_eq5_exploratory/metadata.json`: preprocessing choice, train time range, counts and SHA-256 hashes of both inputs.

If the output directory has already been used, choose a **new** directory; the script refuses to overwrite existing metadata.

**Don't substitute `DAG.npy`** for the learned graph; that would use the ground-truth answer to build the graph. This code rejects the literal `DAG.npy` filename, but scientific integrity still requires correctly identifying the source of any supplied graph.

## E. Inspect output

```bash
python - <<'PY'
import numpy as np
import pandas as pd
w = np.load('outputs/acg_eq5_exploratory/acg_weights.npy')
e = pd.read_csv('outputs/acg_eq5_exploratory/edges.csv')
print('ACG matrix:', w.shape)
print('Learned edge rows:', len(e))
print(e.to_string(index=False))
print('ACG nonzero weights:', np.count_nonzero(w))
PY
```

There are only five learned edges in your five-iteration THP graph, but some conditional weights may be **zero**. Do not invent a particular expected weight or claim a paper score.

## F. Commit one small checkpoint

```bash
git add scripts/build_acg_weights.py tests/test_acg_weights.py setup/STEP23_ACG_WEIGHTS.md
git diff --cached --stat
git commit -m "feat(cloud-rca): implement paper Eq5 ACG edge weighting"
git status --short
```

No files need deletion; data and outputs remain ignored by `.gitignore`.

## G. Still blocked / next scientific step

1. Wireless **55-device physical topology** is missing from the official `18V_55N_Wireless` archive. Do not use the 18×18 event-causality ground-truth DAG as the physical topology.
2. Authors' exact delay interval, time bins, EM likelihood treatment for same/later events, and original split definitions are not specified enough for verified matching of all TTHP results.
3. Suning fault cases and root-cause-node labels are still missing for the paper's GAT/LightGBM localization stage.
4. Ask the paper authors for topology, their implementation or exact derivation, and preprocessed input/splits. Once available, replace the exploratory THP graph with the **verified TTHP-learned graph**, and rerun Eq. (5) with author-specified slice settings.
