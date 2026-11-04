# Paper 3 — Cressant et al., COSI graph design (plus GDN and GraGOD)

**Status:** `COSI/`, `GDN/` and `GraGOD/` are present; the **5G3E source folder, SWaT/Cisco processed data, pre-trained GoogleNews word vectors, and GraGOD datasets/checkpoints are absent**. The bundle contains a small MSL demo in `GDN/data/msl/` only. Do not confuse MSL with any of the three evaluation datasets in the COSI paper.

## A. Install COSI and verify its command-line interface

Start in the root of your repository, with `pyenv` installed (`setup/README.md`).

```bash
bash baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/setup.sh gpu
cd baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI
source .venv/bin/activate
python Start.py --help
python -m unittest discover -s tests -v
```

You should see **three COSI data-validation tests** pass. `Start.py` now accepts `--train`, `--test` and `--adjacency` instead of referring to another researcher's `/home/killian/...` files. For GPU mode you can use `--device cuda`; on CPU use `--device cpu`. In the commands in this document, `COSI/` is your current folder unless otherwise stated.

## B. First working input pipeline check (not COSI research)

The repository includes an unrelated MSL demo under `../GDN/data/msl/`. We provide a deliberately labeled **NON-COSI** correlation-graph fixture only to test CSV and adjacency handling:

```bash
python prepare_smoke_data.py --output smoke_data
python -c "from data_io import load_csv_pair, load_adjacency; t,s,y=load_csv_pair('smoke_data/train.csv','smoke_data/test.csv'); print('train',t.shape,'test',s.shape,'labels',len(y),'edges',load_adjacency('smoke_data/adjacency.csv',list(t.columns)).shape)"
```

Only after the COSI setup completed successfully, you can try a **single-epoch model integration run**:

```bash
python Start.py \
  --train smoke_data/train.csv \
  --test smoke_data/test.csv \
  --adjacency smoke_data/adjacency.csv \
  --device cpu --epochs 1 --batch 32 --slide-win 10 --slide-stride 5 \
  --output-dir pretrained/smoke --report val
```

This uses a **Pearson-correlation graph plus a simple integration-test connectivity rule**, NOT semantic Word2Vec COSI. Its results **must not** be cited as COSI metrics. It has not been end-to-end verified in the offline audit environment because `torch_geometric` was unavailable there.

## C. Obtain real data and dependencies for the paper

The paper evaluates **5G3E**, **Cisco BGP**, and **SWaT**, not MSL. Follow these acquisition routes:

1. **5G3E:** Read and clone https://github.com/cedric-cnam/5G3E-dataset in a separate data directory. For example, from the **repository root** (not COSI/):

   ```bash
   git clone https://github.com/cedric-cnam/5G3E-dataset baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/5G3E-dataset
   ```

   The dataset may require additional downloads and preprocessing following that repository's README. Do not assume the cloned repository contains the exact `dataset_end_total_preprocessed.csv` expected by the COSI notebook. Record its commit, file checksums, normalization, and fault definitions.

2. **SWaT:** Apply for the original dataset through the iTrust Research data portal at https://itrust.sutd.edu.sg/ . Respect the dataset access/license terms. The COSI notebook expects a **preprocessed** CSV with aligned features and suitable label separation, not necessarily the files directly as downloaded.
3. **Cisco BGP:** Locate the dataset source and access method from the Cisco BGP reference in the provided Cressant paper, or request it and the specific 500-Gbps-train/1-Tbps-test preprocessing from the paper authors. There is **no Cisco raw dataset in this archive** and we do not fabricate a download URL.
4. **GoogleNews pretrained Word2Vec:** The `short*.ipynb` notebooks call `KeyedVectors.load_word2vec_format(...)` and expect `GoogleNews-vectors-negative300.bin` (large binary, external). Obtain it from an authorized Word2Vec distribution, check permissions and storage space, decompress if delivered as `.gz`, and point the notebook's `load_word2vec_format` path at it. This model is **not included** and COSI's semantic graph cannot be exactly constructed without equivalent embeddings and training details.
5. Keep original raw datasets outside Git unless licensing expressly permits redistribution. Keep a small README of provenance, raw file hashes, preprocessing version and train/validation/test split definitions.

## D. Produce a real COSI adjacency matrix

There is **no fully automated, tested converter from raw 5G3E/Cisco/SWaT** in the supplied archive. The author supplies Jupyter notebooks. Run these carefully and preserve a copy of the original notebook before changing your experiment settings.

1. In `COSI/` with `.venv` activated, launch Jupyter:

   ```bash
   jupyter lab
   ```

2. In JupyterLab, open the correct notebook: `short.ipynb` for 5G3E, `short_cisco.ipynb` for Cisco, `short_swat.ipynb` for SWaT.
3. Select the `.venv` Python kernel. Find `path_dataset`, `matrix_correlation_path`, `doc_path` (5G3E) and any `load_word2vec_format('/...')` calls. Replace the original author's **absolute paths** with the actual files downloaded and preprocessed on your PC. The existing example path values are placeholders, not runnable files.
4. For 5G3E, **do not assume the 2-row CSV read is a correlation sample**: the original notebook reads two rows to inspect column names and loads its correlation matrix separately from `matrix_correlation_path`. Prepare a **full training-set Pearson correlation matrix** independently and save it as a headerless numeric CSV in exactly the same feature order; for example, with normal-only numeric training data in `data/my_normal_train.csv`, run `python -c "import pandas as pd; d=pd.read_csv('data/my_normal_train.csv'); d=d.drop(columns=['timestamp','attack'],errors='ignore'); d.corr().to_csv('data/corr_matrice.csv',index=False,header=False)"`. In the notebook set `matrix_correlation_path="data/corr_matrice.csv"`. Computing with only two rows is scientifically invalid.
5. Execute cells **one by one, top to bottom** and inspect array shapes. The rows/columns of the saved adjacency must be in **exactly the same order** as the train CSV feature columns.
6. Save a final **numeric, headerless, square NxN CSV** from the computed COSI adjacency matrix. Example in a notebook after computing an NxN matrix named `adj_25np`:

   ```python
   import numpy as np
   np.savetxt('data/cosi_adjacency.csv', adj_25np, delimiter=',', fmt='%.8f')
   ```

   Create the `data/` directory first or choose another existing path. If the computed variable has another name, use that name; do not invent COSI scores. Verify that the matrix is not entirely zero, does not contain NaNs and has one row/column per KPI.
7. Reconstruct corresponding normal-only train CSV and labeled test CSV. **Train:** each column is a KPI, optionally a `timestamp`; no anomalous `attack=1` entries. **Test:** same feature names plus a numeric `attack` column containing only `0` and `1`. If labels are missing, obtain them; do not make up labels. `Start.py` checks and reorders test columns to train order. It cannot guess the matrix order.

## E. Run the actual COSI/GDN baseline after data are ready

While in `COSI/` with its `.venv` active (change paths to match your files):

```bash
python Start.py \
  --train data/my_normal_train.csv \
  --test data/my_labeled_test.csv \
  --adjacency data/cosi_adjacency.csv \
  --edge-policy directed \
  --device cuda --epochs 20 --batch 32 --slide-win 30 --slide-stride 5 \
  --seed 0 --report val --output-dir pretrained/cosi_real_seed0
```

`--edge-policy directed` retains **all** positive off-diagonal directed edges; `--edge-policy upper` reproduces the old helper's upper-triangle-only conversion. **Record the policy and use it consistently**. `--report val` uses the original code's validation-based threshold rule. `--report best` optimizes threshold using **test labels** and is therefore **not a valid held-out estimate**; use it only to compare against a published oracle-style number if explicitly described. `Start.py` now prevents train/test loading from the same hardcoded path and uses a chronological train/validation split with a window-overlap gap.

The graph-specific `COSI/models/GDN.py` **uses the adjacency supplied to it**. The unmodified separate `GDN/models/GDN.py` still builds learned top-K edges. Do not treat them as the same model when comparing graph designs. Store seed, matrix CSV, graph-policy, raw-data preprocessing, epochs, batch and threshold mode beside each result.

## F. Original GDN demonstration, separate from COSI

Open a **new terminal at the repository root** (or `deactivate` to leave COSI) and run:

```bash
bash baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GDN/setup.sh gpu
cd baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GDN
source .venv/bin/activate
bash run.sh cpu msl
```

The `msl` dataset is already included in `GDN/data/msl/`. `run.sh` uses its original configuration (30 epochs). For NVIDIA GPU 0, run `bash run.sh 0 msl`. It should write a model/checkpoint and metrics under this branch, but **this training has not been executed during our audit**. Use CPU first if a GPU error occurs. The old GDN `install.sh` was redirected to the versioned modern setup; do not install upstream CUDA 10.2 wheels with `pip` on your modern system.

## G. GraGOD supporting framework (optional, not the missing COSI dataset)

Open a **new terminal at the repository root**:

```bash
bash baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GraGOD/setup.sh gpu
cd baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GraGOD
source .venv/bin/activate
python models/train.py --help
```

The GraGOD setup uses Poetry 1.8 and its upstream requirements, including a **GitHub-hosted custom PyTorch Lightning branch**. Internet, Git and a compatible upstream branch are required. If installation fails, record the exact error and consult `TROUBLESHOOTING.md`; do not silently substitute a different Lightning version or claim identical experiments.

GraGOD expects processed input files under e.g. `datasets_files/swat/`, with names listed in `datasets/config.py`, and in some configurations a `edge_index.pt` graph; **none of the full paper inputs are present**. Obtain/preprocess these before using:

```bash
python models/train.py --model gdn --dataset swat --params_file models/gdn/params_swat.yaml
```

For full experiment groups the corrected helper `bash train.sh swat` creates `logs_train/` and stops if a model fails. To inspect a trained checkpoint use the original README and adjust `ckpt_folder` in the associated YAML. **Do not run `predict.sh` until all checkpoints and test datasets are actually present.** GraGOD's objective is a general training framework; it is not itself a substitute for COSI graph construction or for matching the Cressant paper's numeric results.

## H. COSI-specific nested GraGOD fork (important for Cisco experiments)

**Do not substitute the standalone `GraGOD/` directory for the nested `COSI/GraGOD/` directory without checking the code.** They differ: the nested fork contains `datasets/cisco.py`, Cisco-specific YAML configurations, and graph CSV loading used for the COSI experiment. It does **not** include a `pyproject.toml`, so its dedicated `setup.sh` uses the closest supplied **standalone GraGOD dependency manifest** as a compatibility starting point; this is not confirmed to be the authors' exact original environment.

To install this separate source fork, start **at the repository root in a fresh terminal**:

```bash
bash baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/GraGOD/setup.sh gpu
cd baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/GraGOD
source .venv/bin/activate
export PYTHONPATH="$PWD"
python models/train.py --help
python -m unittest discover -s tests -v
```

Before training, place your licensed and preprocessed Cisco split files at exactly these paths **within this nested directory**:

```text
datasets_files/cisco/preprocessed_final_short_cisco_train.csv
datasets_files/cisco/preprocessed_new_short_cisco_val.csv
datasets_files/cisco/preprocessed_final_short_cisco_test.csv
datasets_files/cisco/cosi_adjacency.csv
```

The names above originate from the nested code's `datasets/config.py`. Its train/validation/test CSVs and matrix are not provided. You must create them according to the original experiment's preprocessing and train/test traffic selection. `cosi_adjacency.csv` is a **headerless square COSI adjacency**, ordered identically to the model's KPI columns. Its reader uses the **original upper-triangle-only** edge conversion; preserve/record this graph-policy difference when comparing with `COSI/Start.py --edge-policy directed`.

When all four input files exist, the nested Cisco entrypoint is:

```bash
PYTHONPATH=. python models/train.py --model gdn --dataset cisco \
  --params_file models/gdn/params_cisco.yaml \
  --edge_path datasets_files/cisco/cosi_adjacency.csv
```

The nested training code now explicitly **requires** `--edge_path` for Cisco and SWaT experiments, and its graph loader fails when that file is missing or dimensionally inconsistent. This avoids a previously silent fallback to a **fully connected graph**, which would invalidate an experiment labeled COSI. A CLI help and graph-file validation test have been checked in the audit, but model training is pending PyG/Lightning installation and data.

For SWaT in the nested fork, the corresponding data directory is `datasets_files/swat/` and its CSV filenames are in `datasets/config.py`. The same explicit graph rule applies.

## I. Research reproducibility checklist

For each dataset and model, record: raw source and version; train/test traffic settings; feature names and preprocessing; Word2Vec file/version; correlation technique; COSI combination (multiplicative/additive and λ); top-K and threshold; directed/upper edge policy; model architecture; loss/optimizer; seed(s); GPU; threshold selection; pointwise, range, and VUS metrics. Those parameters matter to the paper's reported comparison and must not be silently changed.
