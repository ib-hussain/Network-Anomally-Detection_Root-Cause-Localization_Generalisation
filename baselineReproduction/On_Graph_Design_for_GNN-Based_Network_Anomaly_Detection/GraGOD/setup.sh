#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../../setup/common.sh"
MODE="${1:-gpu}"
setup_python "$HERE"
# Poetry 1.8, not Poetry 2: upstream poetry.lock and README target 1.x.
python -m pip install 'poetry==1.8.5'
export POETRY_VIRTUALENVS_IN_PROJECT=true
cd "$HERE"
# Reuse the .venv created by setup_python; avoid installing global packages.
poetry env use "$VIRTUAL_ENV/bin/python"
poetry install --only main
# The upstream Lightning fork is a separate network dependency; stop on failure.
python -m pip install -r requirements.txt
setup_torch "$MODE"
setup_pyg
python - <<'PYINNER'
import gragod
import pytorch_lightning
import torch_geometric
print('[CHECK] GraGOD/PyTorch Lightning/PyG imports succeed')
PYINNER
cat <<'END'
[SUCCESS] GraGOD package imports completed (not model training).
NEXT: source baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GraGOD/.venv/bin/activate
THEN: follow baselineReproduction/setup/02_COSI.md (GraGOD subsection).
NOTE: The original datasets and training checkpoints are not present.
END
