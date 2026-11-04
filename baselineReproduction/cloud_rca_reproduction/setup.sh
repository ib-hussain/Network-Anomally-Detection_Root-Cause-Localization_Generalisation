#!/usr/bin/env bash
# Deliberately cannot install a baseline that has no supplied source code.
set -euo pipefail
echo 'UNAVAILABLE: source code and original cloud alarm data for Zhou et al. are not in the repository.' >&2
echo 'Read baselineReproduction/setup/03_CLOUD_RCA.md before proceeding.' >&2
echo 'Do NOT treat an independently reimplemented algorithm as the paper reproduction.' >&2
exit 2
