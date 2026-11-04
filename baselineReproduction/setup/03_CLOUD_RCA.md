# Paper 2 — Zhou et al. ACG-LGraphBM cloud root-cause analysis

## Availability: blocked by missing author code AND data

This archive **does not contain** the implementation or the underlying historical cloud-platform alarm-event sequences. The `cloud_rca_reproduction/` folder is a **placeholder for future authorized files**, not a fabricated baseline. Its `setup.sh` exits with status 2 and a clear message rather than inventing a working model.

The attached Zhou et al. paper describes a pipeline with **Time-delay Topological Hawkes Process (TTHP) → directed anomaly correlation graph → weight-threshold neighbor sampling → Graph Attention Network (GAT) → LightGBM classification → case-similarity interpretability**. The paper's feature pipeline uses alarm text (Chinese tokenization/Jieba and TF-IDF/naïve-Bayes labeling), node/system statistics, and component topology. Implementing an approximate alternative without source data or author settings would be an *independent implementation*, **not an exact reproduction**.

## Step 1 — Wait for the authors' source/data response

Ask for the following items if they have not already been requested:

1. Exact source-code commit/release and software licenses.
2. A data access agreement, anonymized alarm-event sequences and component topology.
3. Event-type dictionary, time units, TTHP delay handling and ACG edge thresholds.
4. Train/validation/test splits, ground-truth root-cause nodes, fault-event cases and metrics.
5. Pretrained text classifier/encoding dictionary, package lockfile, trained model checkpoints if available, and official reproduction commands.

There is no need to send another email while waiting if you already requested these. Retain their reply and terms of use for research records.

## Step 2 — When a response arrives

1. Make a new Git branch for importing the source. Keep the original author archive untouched and save its checksum.
2. Put authorized code under `baselineReproduction/cloud_rca_reproduction/` without overwriting original examples or credentials.
3. Keep private data out of version control; document how to request it, do not redistribute it in the research repository.
4. Inspect the source README, dependency versions, experiment entrypoints and I/O schema before installing. Do **not** run unknown shell scripts or unpickling files without inspection.
5. Replace the placeholder `setup.sh` only **after** its actual dependencies and an executable smoke test are confirmed.
6. Create a new changelog entry and Git commit recording the source version and any compatibility changes.
7. Validate TTHP construction, sampling, learned GAT features, LightGBM classifier, metrics and data splits one stage at a time. Then compare the original paper outputs to **your actual computed results**.

## What to do meanwhile

Continue with Paper 1's verified synthetic pipeline and Paper 3's input checks, then obtain original datasets for actual paper reproduction. Do not invent missing data, train a different RCA model under this paper's name, or imply the unavailable paper has been reproduced.
