# Beginner troubleshooting — follow symptoms in order

**Rule:** Copy the complete last error, command, current directory (`pwd`), Python (`which python`, `python --version`), and GPU state (`nvidia-smi`) into your experiment log before changing anything.

## 1. `pyenv: command not found`

Complete `setup/README.md` Step 2, close and reopen Terminal, and run `pyenv --version`. Don't run a `setup.sh` with `sudo`. If Python compilation fails, confirm `build-essential`, `libssl-dev`, `zlib1g-dev`, `libreadline-dev`, `libsqlite3-dev`, and `liblzma-dev` are installed. `pyenv install -s 3.10.13` will not replace system Python.

## 2. `.venv/bin/activate: No such file or directory`

You are either in the wrong folder or the previous setup script stopped with an error. Run `pwd`, check `ls .venv/bin/activate`, then rerun the correct baseline's `setup.sh`. Each baseline has a **different `.venv`**.

## 3. `ModuleNotFoundError: torch_geometric` or `torch`

First: `which python`; it should point inside that baseline's `.venv/bin/python`. Then rerun the corresponding baseline setup. Avoid mixing `sudo pip`, global Python, conda and pyenv venvs. Our setup pins **torch 2.4.1 + torch-geometric 2.6.1** for GDN/COSI. PyG 2.6 supports core `MessagePassing` without requiring every optional compiled extension. If compiled operator errors persist, see https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html and match the exact PyTorch/PyG/CUDA wheel pair; do not randomly mix wheel versions.

## 4. `torch.cuda.is_available()` returns False

Run `nvidia-smi` on Ubuntu. If it fails, diagnose driver/WSL GPU forwarding first. If it works, print `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. A `+cpu` build cannot use the GPU; rerun `setup.sh gpu`. NVIDIA driver *reported* CUDA 13.0 does not mean you should install ancient CUDA 10.2 or that CUDA 12.1 PyTorch wheels cannot run.

## 5. `CUDA out of memory`

Close other GPU jobs (`nvidia-smi`), use smaller `--batch` where the CLI provides it, use the smoke config or a smaller dataset for checks, and restart the run. Do not silently change paper results without logging the new batch size. Your ~20 GB GPU is not unlimited.

## 6. `No such file or directory` for dataset paths or GoogleNews word vectors

The full **paper datasets are not in the archive**. Follow `01_TOPOKPI.md` or `02_COSI.md` data-acquisition steps. A clone of 5G3E may not include the exact preprocessed CSV; check its README. COSI notebooks contain original author's absolute paths; edit those cells to match *your* obtained data. A CSV filled with random fabricated labels is not a reproduction.

## 7. `INPUT ERROR: Adjacency shape ... expected (N,N)`

In COSI, feature names in the **normal train CSV** define N and row/column order. The adjacency CSV must be exactly NxN, no header, no name column, numeric, and built from the exact same ordered feature list. `attack` and `timestamp` are not graph features. The `data_io` input checks reject misaligned data rather than guessing what to drop.

## 8. COSI results look excellent when using `--report best`

That mode searches a **test-label-optimized** threshold, inflating held-out comparisons. Use `--report val` unless you are explicitly reproducing an oracle comparison described in the paper. Always report threshold source, dataset and seed.

## 9. Poetry / Lightning fails while installing GraGOD

The GraGOD setup depends on `poetry==1.8.5`, its lock file and an external GitHub Lightning fork. First verify `git --version` and internet access from the same machine. If the fork/ref is deleted or cannot be fetched, **record the exact failure** and do not claim GraGOD reproduction. Using official Lightning instead would be a *different dependency configuration* requiring equivalence testing.

## 10. COSI nested GraGOD warns about a missing graph / fully connected fallback

There are **two different GraGOD code copies**. The COSI-nested one has Cisco code and requires `--edge_path` when running its training script on Cisco/SWaT. The supplied file must be headerless NxN, aligned with KPI order, and exist at that location. If it does not exist, the patched program stops instead of silently training on the wrong graph. The nested `setup.sh` borrows the standalone project's dependency manifest because the nested fork has no pyproject; installation is a compatibility hypothesis, not an exact original environment claim.

## 11. Mininet `Permission denied`, OVS or host-network errors

Actual fault injection requires Linux, working Mininet/Open vSwitch and root privileges. Start with `--dry-run`. For a real run, use the command using `sudo env PYTHONPATH="$PWD/src" "$PWD/.venv/bin/python" ...` in `01_TOPOKPI.md` so that the root process still uses the project's environment. Use an isolated test network: Mininet changes host networking.

## 12. Unit tests pass but paper numbers differ

This is expected with synthetic/other datasets. Compare raw data, event labels, feature order, graph settings, train/validation/test time boundaries, random seeds, metric aggregation, and threshold choice. The reported CSV files shipped in Paper 1 are reference numbers, **not measurements of the run you just made**.
