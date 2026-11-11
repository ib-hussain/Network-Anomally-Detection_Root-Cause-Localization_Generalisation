# Step 22 — Paper-faithful TTHP specification and reproduction audit

**Reference:** B. Zhou et al., *Root cause analysis of cloud platform faults based on anomaly correlation graph and graph neural network*, Engineering Applications of Artificial Intelligence **174 (2026), 114451**.  
**Purpose:** audit **what the paper says**, separate it from our engineering experiments, and record blocked decisions **before** writing an alleged TTHP trainer.  
**Scope:** Wireless causal learning first; Suning RCA classification is a separate task.  
**IMPORTANT:** This document does **not** claim reproduction, invent a physical topology, or claim the paper's optimizer is implemented.

## 1. Source of truth and terminology

- **Physical topology**: `G_N=(N,E_N)`, containing device/component nodes and connections. **NOT** the 18×18 ground-truth event-type DAG.
- **Event causality graph**: `G_V=(V,E_V)`; one vertex per abnormal event **type**. Output of THP/TTHP learning; at this point directed and **unweighted**.
- **Anomaly correlation graph (ACG)**: directed **weighted** graph obtained by assigning conditional-probability weights to edges of learned `G_V` (Eq. 5, Algorithm 1).
- **RCA task**: root-cause **node** ranking from Suning fault cases using sampled topology, GAT, and LightGBM. Not equivalent to Wireless causal edge F1c.

| Component | Exact paper location | Paper-defined requirement | What exists locally | Status |
|---|---|---|---|---|
| Input and physical graph | §3.1, p.4 | Events `(node, type, time)` and physical `G_N` | `Alarm.csv` transformed to `event,timestamp,node`; no verified physical graph | **BLOCKED for exact THP/TTHP** |
| THP intensity | Eq. (1), p.4 | Baseline `mu_v`, topology powers `Â^k`, coefficients `alpha_{v',v,k}`, earlier-time kernel `kappa(t-t')` | gCastle `TTPM` only tested with `max_hop=0` | **Exploratory baseline** |
| Penalized objective | Eq. (2), p.4 | `LB = Const - p log(m)/2 + sum_{v,t,n} [ -lambda_v(n,t) Delta_t + X_{n,v,t} log(lambda_v(n,t)) ]` | Our prototype: Poisson NLL + L2; not Eq. (2) | **Not implemented** |
| Parameter-count penalty | Below Eq. (2), p.4 | Paper prints `p = |V| + K |E_V|`; `m` = number of observed events | No BIC-like graph score implemented for TTHP | **Not implemented; index convention needs checking** |
| Parameter fitting / graph search | §3.1, p.4; Algorithm 1 p.8 | EM for candidate graph parameters; edge **add / delete / reverse**; retain graph with best `LB` until no improvement | Past-only L-BFGS-B numerical prototype; no edge search | **Not paper estimator** |
| Delay-aware intensity | Eq. (4), §4.2 p.7 | Replace earlier-only set with `T^-_{t+Delta_t_l}`; use `kappa(|t-t'|)`; includes some same/later timestamps | `tthp_delay_kernel.py` single-device event-pair diagnostic | **Kernel prototype only** |
| Graph-to-ACG weights | Eq. (5), pp.7–8; Table 1 | Slice time, set binary event-type presence per slice, and weight learned edge `vi -> vj` by `count(vj | vi) / count(vi)` | Not implemented | **Not implemented** |
| Noise filtering | §4.3 pp.8–9 | Map alarm-type causal weights onto physical neighbor links; threshold sampling | No Suning case/topology available | **BLOCKED** |
| Node features and classifier | §§4.4–4.5 pp.9–10 | Eight GAT heads × eight hidden features = 64; four statistical features, then LightGBM | Not implemented | **BLOCKED by labeled cases** |
| Evaluation | §§5.1–5.3 pp.11–14 | Wireless: directed event causality F1c, Recallc, Precisionc; Suning: root cause Recall/Precision/F1/Pre@K | Exploratory 18×18 evaluation only | **Not matched to paper protocol** |

## 2. Equations and non-negotiable mathematical checks

### Eq. (1): original topology-aware THP intensity, p.4

`lambda_v(n,t) = mu_v + sum_{v' in PA_v} sum_{n' in N} sum_{k=0}^K alpha_{v',v,k} (Â^k)_{n',n} sum_{t' in T^-_t} kappa(t-t') X_{n',v',t'}`.

- Paper defines `Â = D^(-1/2) A D^(-1/2)` for device topology; `T^-_t` contains earlier time points `t'<t`.
- `k=0` is included in the printed equation. If using zero-hop only, this is a **restriction of the model**, not paper-equivalent topology awareness.
- `X_{n,v,t}` counts events of type `v` at node `n` in the time interval described immediately before Eq. (1); inspect units/observation grid before fitting.

### Eq. (2): penalized log-likelihood, p.4

`LB(G_V,Theta;X,G_N) = Const - [p log(m)]/2 + sum_{v,t,n} [-lambda_v(n,t) Delta_t + X_{n,v,t} log(lambda_v(n,t))]`.

- The penalty scales with candidate **graph complexity**, not simply L2 on `alpha`. The paper prints `p=|V|+K|E_V|`, although Eqs. (1)/(4) sum `k=0..K`. **Do not silently replace the paper's printed convention with `(K+1)|E_V|`**. This needs a documented interpretation.
- Eq. (2) is an approximate **binned** process objective. It is not the same objective as `fit_wireless_past_only.py` (Poisson count NLL with `gammaln` + L2 and no graph penalty).
- Parameter positivity, integrated rate approximation, and topology normalization must be tested explicitly.

### Eq. (4): proposed delay-aware TTHP intensity, p.7

`lambda_v(n,t) = mu_v + sum_{v' in PA_v} sum_{n' in N} sum_{k=0}^K alpha_{v',v,k} (Â^k)_{n',n} sum_{t' in T^-_{t+Delta_t_l}} kappa(|t-t'|) X_{n',v',t'}`.

- According to §4.2, events collected **at the same time or later** within an acquisition-delay window may contribute. The kernel is based on **absolute** time difference.
- The paper does **not** give a reproducible numeric value for `Delta_t_l` in this equation, nor an exhaustive account of how **current-bin self-counts**, simultaneous events, or **future observation dependence** are handled in EM and the `-lambda Delta_t` term.
- The printed set `T^-_{t+Delta_t_l}` suggests `t'<t+Delta_t_l`; our exploratory kernel used an **inclusive** `t'<=t+delay` boundary and excluded the record itself. These are **implementation decisions, not verified reproductions**.
- **Critical**: using events at/after `t` in a conditional intensity makes ordinary past-filtration point-process likelihood assumptions questionable. Eq. (2) is referenced by the paper after Eq. (4), but the necessary estimation details are not supplied in the PDF. Resolve or explicitly qualify the treatment before claiming a mathematically faithful estimator.
- Do **not** interpret `end_timestamp - start_timestamp` as the collection-delay parameter; these columns represent recorded alarm intervals, not an identified collection-latency measurement.

### Eq. (5): ACG directed weight, p.8

The paper computes binary event-type presence **per time slice** (Table 1), then attaches to an existing learned directed edge `v_i -> v_j` a conditional co-occurrence fraction:

`weight_ij = count_{v_j/v_i} / count_{v_i}`.

Interpretation to validate from the authors' code: `# slices with both (i,j) / # slices with i`. **Only weight edges already selected by TTHP**. Time-slice width is not fixed unambiguously in the PDF; do not set one solely from our exploratory 512-bin experiment. Edge orientation matters: `weight_ij` is not generally equal to `weight_ji`.

### Algorithm 1: graph search, p.8

1. Start `G_V*` as an empty **event-type** graph; initialize parameters.
2. Generate neighbors by **one edge addition, deletion, or reversal**.
3. For each candidate graph, fit `Theta` with EM (until convergence) and calculate **Eq. (2)** score.
4. Choose highest-scoring neighbor; repeat while score improves.
5. Compute Eq. (5) conditional weights for the resulting directed causal edges.

The paper does not fully specify the EM update formulas, candidate edge admissibility constraints, initialization distributions, numerical tolerances, or graph-search tie-breaking. A strict reproduction needs author code or explicitly documented assumptions and equivalence tests.

## 3. What our existing experiments prove — and do not prove

| Local code / experiment | Verified purpose | **Not** evidence of |
|---|---|---|
| `prepare_wireless.py` | Reformat alarm logs; preserve 18 event IDs and 55 physical device identities | Availability of full device topology |
| `train_thp_smoke.py`, `train_thp_18types.py`, `train_thp_18types_iter5.py` | gCastle THP graph inference with **identity topology and max_hop=0** | Reproducing paper's topology-aware THP or TTHP result |
| `inspect_tthp_delay.py` | Counts candidate event pairs within chosen time windows | Actual causal-edge inference |
| `tthp_delay_kernel.py` | Toy same-device exponential pairwise features; no learned coefficients | Full Eq. (4) topology summation or optimizer |
| `tthp_toy_likelihood.py` | Binned Poisson likelihood test; **contains present-bin counts** | Valid conditional point-process likelihood for delay-aware TTHP |
| `fit_past_only_thp.py` | Synthetic L-BFGS-B gradient and parameter recovery | Paper's EM fitting |
| `fit_wireless_past_only.py` | Exploratory 512-bin, same-device, past-only Poisson fit; optimizer **did not converge at 150 iterations** | Paper TTHP model or causal graph score |

**Pause speculative fitting**: another 500-iteration L-BFGS-B run is optional optimizer diagnostics; it is **not required** before implementing or auditing the paper-defined estimator. Do not tune against the test DAG.

## 4. Data and evaluation protocol from the PDF

- **Wireless (§5.1.1, p.12)**: the paper states **34,839** records, 18 event types, and 55 devices. The user's verified public archive parses as **34,838 CSV data rows**, 18 event types, 55 devices, with a 18×18 ground-truth event DAG. The one-record discrepancy is **unresolved**; do not fabricate a row or silently alter the paper's count. Check whether the paper counted the CSV header or used another release.
- **Wireless graph metric (§5.2, p.13)**: directed **Precisionc, Recallc, F1c**, with Table 5 mean ± SD across **10 random splits**. §5.2 reports **7:3 train/test**, but does not provide the random seeds or exact split index files.
- **Published Table 5, p.13**: THP `F1c 0.565±0.020`; TTHP `F1c 0.664±0.012`. These are *paper references*, not targets to overfit by changing parameters on the evaluation DAG.
- **Suning (§5.1.1, pp.11–12)**: **100 labeled fault-case groups**, **84,304 alarm records**, 100 component nodes across 10 systems, original physical topology; this dataset is required for the supervised RCA pipeline. The full source and corresponding author code are **not present**.
- **RCA tuning (§5.4, p.15)**: weight threshold **0.7**, GAT **8 heads**, LightGBM learning rate **0.03**, max leaves **15**, and maximum feature bins **255**. These are later-stage settings, **not** missing TTHP delay/EM hyperparameters.

## 5. Known blockers and decision register

| ID | Required to claim faithful replication | Source / missing detail | Action |
|---|---|---|---|
| B01 | 55×55 physical connectivity graph and direction semantics | Paper Eq. (1), `G_N`; public `Alarm.csv`/`DAG.npy` do not contain this graph | Request topology or a documented extraction recipe from authors |
| B02 | Time discretization and meaningful units for `Delta_t`, `Delta_t_l`, `kappa` | §3.1 & §4.2; PDF gives formulation but insufficient numeric settings | Ask for configs and preprocessing script |
| B03 | EM updates, initialization, parameter constraints, neighbor search and stopping | §3.1, Algorithm 1; no complete runnable implementation supplied | Request code/parameters; if independently implemented, label as **reimplementation** |
| B04 | Correct future/same-bin treatment in the proposed Eq. (4) likelihood | Eq. (2) and Eq. (4) used together; observation-filtration ambiguity | Formally derive/test an objective; request clarification before training |
| B05 | Exact 7:3 split protocol and 10 seeds | §5.2, Table 5 | Record deviation; avoid selecting hyperparameters on held-out DAG |
| B06 | 34,839 vs 34,838 event count difference | §5.1.1 vs downloaded CSV | Inspect source provenance; do not add data artificially |
| B07 | Suning alarm text, fault cases, topology, root-cause node labels and data permissions | §5.1.1; paper says data available on request | Obtain authorized files before claiming ACG-LGraphBM RCA reproduction |

## 6. Immediate actions, in order

1. Run `python -m scripts.audit_tthp_inputs` (new diagnostic packaged alongside this document) to verify local Wireless artifacts and print blocking topology/data findings. Its status is **blocked**, not failure of preprocessing.
2. Keep the existing `.venv-thp`, code, outputs, and Git history. Do not delete, overwrite, or merge experimental models into production implementation.
3. Request **physical topology**, **exact Δt_l / time grid**, **EM and graph-search code/configs**, and **split seeds** from authors; ask for the Suning dataset under its access terms.
4. Only after a decision is recorded for B01–B05, implement a separate `tthp_paper/` code path with unit tests reproducing Eqs. (1), (2), (4), and Algorithm 1. Mark any guesses `ASSUMPTION`, not paper-provided facts.
5. Reproduce Wireless causal F1c first, then Suning root-cause metrics. Maintain separate baseline, independence-test, and ground-truth metrics.

**Change control:** This file and its accompanying read-only audit script/test are a documentation-and-validation checkpoint. They **do not modify existing experiments** and should be committed on the existing `exp/tthp-parameter-fitting` branch before writing new training code.
