#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../setup/common.sh"
MODE="${1:-gpu}"
setup_python "$HERE"
setup_torch "$MODE"
python -m pip install -e "$HERE"
cd "$HERE"
python -m unittest discover -s tests -v
python -m topokpi.cli --help >/dev/null
cat <<'END'
[SUCCESS] TopoKPI dependencies and unit tests completed.
NEXT: source baselineReproduction/topology_kpi_reproduction/.venv/bin/activate
THEN: follow baselineReproduction/setup/01_TOPOKPI.md
NOTE: This does NOT reproduce paper results without the paper's Mininet dataset.
END
