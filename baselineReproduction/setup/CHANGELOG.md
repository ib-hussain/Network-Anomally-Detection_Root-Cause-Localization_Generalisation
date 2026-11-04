# Engineering change log — baseline alignment (2026-11-04)

Each entry is scoped for individual Git commits. No original research model was silently replaced and no source-paper dataset was invented. Affected files are listed in `GIT_WORKFLOW.md`.

## Change 1 — Reproducibility guardrails and safe setup

- Replaced the baseline root README (which only contained clone links) and the empty repository root README with navigation and truthfully labeled readiness.
- Added explicit beginner instructions, GPU/CPU modes, `pyenv` 3.10.13 setup, isolated per-baseline virtual environments and shared setup helpers.
- Updated `.gitignore` for generated local environments/caches/experiment outputs. Did **not** remove any original files.

## Change 2 — Paper 1 environment and validation

- Added `topology_kpi_reproduction/setup.sh` using Python 3.10.13, PyTorch 2.4.1 cu121 (or CPU), editable install and existing unit-test suite.
- **Observed during audit:** 12/12 bundled unit tests pass; a CPU synthetic 12-node, 320-step demo trained for 2 epochs and evaluated successfully. No changes were necessary to the scientific model implementation.
- **Not verified:** NVIDIA GPU execution, author Mininet dataset, large campaign results or Train-Ticket public benchmark download.

## Change 3 — Paper 3 COSI input integrity and graph usage

- Reworked COSI `Start.py` with explicit normal train, labeled test and adjacency paths; removed fixed `/home/killian` paths, identical train/test input, implicit data truncation and unsafe checkpoint device assumptions.
- Added `data_io.py` to reject missing/bad data, nonnumeric values, invalid labels and dimension mismatches. Test KPI columns are reordered to match train KPI columns **without** guessing adjacency order.
- Uses a chronological validation split with overlap gap and `--report val` default. The legacy test-label oracle mode remains explicit as `--report best` with a warning.
- Removed a **calculation of learned top-K edges that COSI's supplied-edge GDN forward pass never used**; it still uses the supplied edges and the same encoder/prediction head. The original GDN branch remains distinct.
- Added three passing `data_io` tests and `prepare_smoke_data.py` producing a prominently labeled **NOT-COSI** MSL/Pearson integration fixture.
- In `short.ipynb`, corrected an output path that attempted to write into the filesystem root. Preserved the original 2-row feature-name preview, since the notebook loads a separately computed full correlation matrix. Notebook paths and external Word2Vec/data prerequisites remain user-specific and are documented.
- **Not verified:** full COSI/GDN GPU training with PyG; paper datasets, pretrained model, full notebook execution, or reproduction of any published score.

## Change 4 — GDN/GraGOD support

- Added GDN setup with modern separately pinned PyTorch/PyG, checking bundled MSL data and imports. Redirected unsafe upstream CUDA 10.2 installer to this setup. Made `GDN/run.sh` run from any current directory and validate arguments/data paths.
- Added GraGOD setup targeting its Poetry 1.x Git-hosted dependencies. Fixed `GraGOD/train.sh`'s missing logs directory and failure propagation; made `predict.sh` resolve its working directory.
- Repaired the **nested COSI/GraGOD fork** separately: it has no pyproject but does have its own Cisco dataset/graph handling. Added a dependency-compatibility setup, removed an author-specific import path, replaced personal dataset paths with project-relative paths, made explicit graph file requests fail instead of silently selecting a fully connected graph, and added two passing graph validation tests.
- Fixed a stray un-commented `%%` in a standalone GraGOD exploratory research script so all supplied Python sources compile.
- **Not verified:** GDN/GraGOD model training and third-party dependency installation on user hardware. Neither fork includes the required data/checkpoints.

## Change 5 — Unavailable paper + operational documentation

- Added `cloud_rca_reproduction/setup.sh` as an **intentional informative failure**, not a placeholder fake training implementation.
- Added detailed original-data acquisition, experiment provenance, troubleshooting, Git commit and rollback instructions. No private data is bundled.

## Known outstanding issues / blockers

1. Paper 1 original Mininet KPI datasets and exact protocol matching remain pending author access or independent data collection.
2. Paper 3 full 5G3E/Cisco/SWaT prepared data and pretrained GoogleNews Word2Vec files remain absent; notebooks require those paths and a compatible preprocess.
3. Paper 3 GraGOD depends on an external Lightning fork and missing processed datasets; this was not installed in the offline audit.
4. Paper 2 code and original alarm data are pending the authors' reply; there is no honest way to claim its reproduction yet.
5. Modernizing old GDN/COSI to PyG 2.6.1 is a compatibility attempt and requires on-machine runtime verification before scientific benchmarking.

**No deletions are required** by this patch. Historical baseline ZIP and upstream files remain in place.
