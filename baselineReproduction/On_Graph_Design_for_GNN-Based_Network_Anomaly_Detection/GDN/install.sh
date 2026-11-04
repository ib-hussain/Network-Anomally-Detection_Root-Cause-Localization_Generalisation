#!/usr/bin/env bash
# Avoid the upstream installer targeting unavailable PyTorch 1.5/CUDA 10.2 wheels.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/setup.sh" "${1:-gpu}"
