#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../../setup/common.sh"
MODE="${1:-gpu}"
setup_python "$HERE"
setup_torch "$MODE"
python -m pip install 'numpy==1.26.4' 'pandas==2.2.3' 'scikit-learn==1.5.2' 'scipy==1.14.1' 'matplotlib==3.9.2' 'seaborn==0.13.2' 'gensim==4.3.3' 'nltk==3.9.1' 'plotly==5.24.1' 'jupyterlab==4.2.5' 'ipykernel==6.29.5'
setup_pyg
cd "$HERE"
python -m unittest discover -s tests -v
python Start.py --help >/dev/null
cat <<'END'
[SUCCESS] COSI environment imports and independent data-validation tests completed.
NEXT: source baselineReproduction/On_Graph_Design_for_GNN-Based_Network_Anomaly_Detection/COSI/.venv/bin/activate
THEN: follow baselineReproduction/setup/02_COSI.md.
NOTE: Full paper reproduction still requires original data and COSI graph CSV.
END
