#!/usr/bin/env bash
# The COSI-specific GraGOD fork has no pyproject.toml. Reuse sibling dependency
# manifest, but run this fork from its own folder and preserve its source files.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../../../setup/common.sh"
MODE="${1:-gpu}"
setup_python "$HERE"
python -m pip install 'poetry==1.8.5'
# Sibling GraGOD is the closest supplied dependency manifest for this fork.
# This is a compatibility hypothesis: its locked versions are NOT confirmed
# to be identical to those used for all COSI journal experiments.
poetry -C "$HERE/../../GraGOD" export --only main --without-hashes --format requirements.txt --output "$HERE/.dependencies-export.txt"
python -m pip install -r "$HERE/.dependencies-export.txt"
setup_torch "$MODE"
python -m pip install -r "$HERE/requirements.txt"
setup_pyg
cd "$HERE"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
python models/train.py --help >/dev/null
cat <<'END'
[SUCCESS] Nested COSI/GraGOD environment CLI help ran.
NOTE: This fork requires COSI-specific data and explicit --edge_path.
NEXT: follow baselineReproduction/setup/02_COSI.md, section H.
END
