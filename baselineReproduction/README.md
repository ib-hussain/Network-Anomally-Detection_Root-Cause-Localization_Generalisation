# Baseline reproduction — start here

This directory contains the **three papers being evaluated**, but **only two supplied code implementations**. Follow [setup/README.md](setup/README.md) in order.

| Paper | Local code | Status | Beginner guide |
|---|---|---|---|
| Luo et al. (topology-aware KPI anomaly detection and RCA) | `topology_kpi_reproduction/` | Code + unit tests + synthetic demonstration; **original Mininet data missing** | `setup/01_TOPOKPI.md` |
| Cressant et al. (COSI graph design) | `On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/`, `GDN/`, `GraGOD/` | Graph notebooks and trainers; **paper datasets and pretrained word vectors missing** | `setup/02_COSI.md` |
| Zhou et al. (ACG-LGraphBM cloud RCA) | `cloud_rca_reproduction/` | **No author code or cloud alarm dataset included**; setup is an informative, deliberate stop | `setup/03_CLOUD_RCA.md` |

Do not report a synthetic or MSL demonstration as a published-results reproduction. Do not treat numbers already in `topology_kpi_reproduction/data/reported/` as metrics measured by your own run. Preserve the code and dataset provenance and record paper/seed/config/checkpoint for every experiment.

Original upstream resources (for provenance; no automatic clones by the setup scripts):
- https://github.com/cedric-cnam/5G3E-dataset
- https://github.com/Killian-cressant/COSI
- https://github.com/d-ailin/GDN
- https://github.com/GraGODs/GraGOD

See `setup/VALIDATION_REPORT.md` for tests and untested claims, `setup/CHANGELOG.md` for modifications, `setup/GIT_WORKFLOW.md` for incremental commits, and `setup/TROUBLESHOOTING.md` for failures.
