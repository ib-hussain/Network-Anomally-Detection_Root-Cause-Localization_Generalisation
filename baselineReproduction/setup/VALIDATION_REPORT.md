# Verification report — 2026-11-04

This report separates actual tool-executed results from unverified installation assumptions. The audit ran inside a **CPU-only, Python 3.13.5 / PyTorch 2.10.0+cpu** container, not your Ubuntu RTX 4000 Ada machine. We did **not** install `torch_geometric`, `pyenv`, Poetry or the remote data in that container.

| Item | What was run | Outcome | What this proves |
|---|---|---|---|
| Original TopoKPI Python tests | `PYTHONPATH=src python -m unittest discover -s tests -v` | **12 passed** | Built-in Python model and utility tests succeed in CPU container |
| TopoKPI synthetic data | `python -m topokpi.cli generate-demo --nodes 12 --steps 320 --output ...` | **Passed** | Synthetic NumPy dataset generated correctly |
| TopoKPI end-to-end | `python -m topokpi.cli train ... --config configs/smoke_test.json --device cpu` then `evaluate` | **Passed**, 2 epochs, JSON detection/RCA outputs | End-to-end CPU synthetic pipeline works; **not paper reproduction** |
| COSI data I/O | `python -m unittest discover -s tests -v` in `COSI/` | **3 passed** | CSV labels/order, graph dimensions, invalid input checks |
| COSI demo input prep | `python prepare_smoke_data.py --output ...`; then `data_io.load_csv_pair`, `load_adjacency` | **Passed** | Non-paper MSL/Pearson demo input files can be built and parsed |
| COSI CLI help | `python Start.py --help` | **Passed** | Argument parsing works without the original author data |
| COSI-nested GraGOD input checks | `PYTHONPATH=. python -m unittest discover -s tests -v` | **2 passed** | Missing or mis-sized explicitly requested graph files fail safely |
| Source compilation | `python -m compileall -q baselineReproduction` | **Passed** | Every `.py` file has valid Python syntax under Python 3.13 |
| Bash syntax | `bash -n` on new setup scripts and edited run/train/predict shell scripts | **Passed** | Edited shell scripts parse correctly |
| CUDA and user hardware | No GPU access in audit container | **NOT TESTED** | Run `nvidia-smi` + your `.venv` `torch.cuda.is_available()` |
| PyG/Poetry remote installation | No external package download in audit container | **NOT TESTED** | Modern compatibility pins are a best-effort install recipe only |
| COSI/GDN full training | Requires PyG; original data not present | **NOT TESTED** | Do not report metrics as reproduced |
| Paper 1 real Mininet/Train-Ticket | Raw data absent | **NOT TESTED** | Only synthetic pipeline confirmed |
| Paper 2 ACG-LGraphBM | Source and alarm data absent | **BLOCKED** | No false implementation created |
| Paper 3 true COSI graph reproduction | Source evaluation data/Word2Vec absent | **BLOCKED** | Not run with paper inputs |

The fixed `GraGOD/research/datasets/20250217_metrics_analysis.py` contained a stray bare `%%` at line 130; that prevented even Python compilation before correction. The whole Python source tree compiles after correction. Compilation **does not imply dependency resolution or a successful model run**.

**Required next verification on user machine:** run each `setup.sh` and record its complete log, `python --version`, `pip freeze`, `nvidia-smi`, GPU visibility, paper dataset hashes and actual train/evaluate outputs. Treat any failed installation as an unresolved blocker, not as success.
