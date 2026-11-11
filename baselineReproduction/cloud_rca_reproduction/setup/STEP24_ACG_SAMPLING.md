# Step 24 — ACG weight-threshold sampling (Section 4.3)

## Status

The source is Zhou et al. (2026), Section 4.3. It first forms a **case-specific topological subgraph of nodes with alarms**, then removes topological edges whose alarm-type ACG weight is below a threshold. **Do not** present filtered event-type edges as the paper's case-specific alarm correlation subgraph (ACSG).

The paper's directed lookup is easy to reverse accidentally:

- Physical topology has edge **node_i → node_j**.
- Node i's most frequent alarm type is `vi`; node j's is `vj`.
- Look up **ACG[vj, vi]** (reverse alarm direction, explicitly described in Section 4.3).
- Keep that *node* edge when `ACG[vj,vi] >= threshold`.
- Nodes with no alarm within the case are absent. For ties among most-frequent alarm types, our implementation selects the smallest event-type ID; **tie treatment is an implementation assumption**.

The paper identifies **0.7** as a good threshold on its **Suning fault-localization dataset**, not necessarily optimal on Wireless. The paper is internally unclear about behavior at exactly threshold 0; this implementation intentionally requires a positive threshold.

## A. Run the available Wireless **event-type preview only**

Start inside `baselineReproduction/cloud_rca_reproduction/` with `.venv-thp` activated:

```bash
python -m unittest discover -s tests -v
python -m scripts.sample_acg_neighbors \
  --mode preview \
  --acg outputs/acg_eq5_exploratory/acg_weights.npy \
  --threshold 0.7 \
  --output outputs/acg_threshold07_preview
cat outputs/acg_threshold07_preview/retained_edges.csv
```

Expected from the **five weights already reported**: **4 event-type edges** (6→11, 7→6, 7→11, 14→11) meet the threshold; 14→10 (0.377593) does not. This is a derived preview, NOT a network-node ACSG or paper reproduction. If the output directory already exists, choose a new one rather than overwriting it.

## B. Case-specific node sampling (only when real inputs are available)

The complete model **requires** an authorized directed physical topology for the cloud network, with N nodes, and **a single fault case** with observed alarms mapped to those N node IDs. Neither is available in the current public Wireless release. The script does not invent them.

- `topology.csv`: **headerless**, N×N numeric 0/1 adjacency, row i → column j; no self-loops. Node IDs must be consecutive integers 0 to N−1 matching these rows.
- `case_alarms.csv`: CSV **with header** `node_id,event_type`. Include one row for each observed alarm; repeated occurrences are counted. Both fields use zero-based integer indices. This file represents **one fault case only**.
- `acg_weights.npy`: the directed event-type ACG saved in Step 23; do not substitute `DAG.npy`.

Once those real, matched case-specific inputs exist:

```bash
python -m scripts.sample_acg_neighbors \
  --mode case \
  --acg path/to/acg_weights.npy \
  --topology path/to/topology.csv \
  --alarms path/to/case_alarms.csv \
  --threshold 0.7 \
  --output outputs/first_real_case_sample
```

The program produces `retained_edges.csv` and `metadata.json`. This is **only the sampling stage**, not GAT or RCA training. The Wireless graph from Step 23 was learned using the full dataset; it is not leakage-free for held-out evaluation. For true reproduction obtain the authors' topology, case labels, preprocessing and split protocols.

## C. Incremental Git checkpoint

From the same folder, only after tests and preview pass:

```bash
git add scripts/sample_acg_neighbors.py \
  tests/test_acg_neighbor_sampling.py \
  setup/STEP24_ACG_SAMPLING.md
git diff --cached --stat
git commit -m "feat(cloud-rca): implement topology-aware ACG threshold sampling interface"
git status --short
```

No existing files need deletion or modification. `outputs/` is intentionally excluded from Git.
