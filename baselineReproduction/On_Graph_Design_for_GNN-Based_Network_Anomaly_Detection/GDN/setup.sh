#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../../setup/common.sh"
MODE="${1:-gpu}"
setup_python "$HERE"
setup_torch "$MODE"
python -m pip install 'numpy==1.26.4' 'pandas==2.2.3' 'scikit-learn==1.5.2' 'scipy==1.14.1' 'matplotlib==3.9.2'
setup_pyg
cd "$HERE"
python - <<'PYINNER'
import pandas as pd
from pathlib import Path
assert Path('data/msl/list.txt').exists()
a = pd.read_csv('data/msl/train.csv')
b = pd.read_csv('data/msl/test.csv')
assert 'attack' in b.columns and len(a)>30 and len(b)>30
print('[CHECK] Bundled MSL demo train/test present:', len(a), len(b))
from models.GDN import GDN
print('[CHECK] GDN code imports')
PYINNER
cat <<'END'
[SUCCESS] GDN environment imports and example-data checks completed.
NEXT: source baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/GDN/.venv/bin/activate
THEN: follow baselineReproduction/setup/02_COSI.md (GDN subsection).
NOTE: End-to-end training was NOT executed by this installer.
END
