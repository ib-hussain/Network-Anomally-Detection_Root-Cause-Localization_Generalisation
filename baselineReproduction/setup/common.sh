#!/usr/bin/env bash
# Shared safe setup functions. Must be sourced, not executed directly.
set -euo pipefail

setup_python() {
  local project_dir="$1" version="${PYTHON_VERSION:-3.10.13}"
  if ! command -v pyenv >/dev/null 2>&1; then
    echo 'ERROR: pyenv is missing. Follow baselineReproduction/setup/README.md, Step 2.' >&2
    return 1
  fi
  if [[ "$(uname -s)" != Linux ]]; then
    echo 'ERROR: These scripts target Ubuntu Linux. On Windows use a Linux/WSL environment.' >&2
    return 1
  fi
  echo "[SETUP] Ensuring Python ${version} with pyenv (may download/build Python)..."
  pyenv install -s "$version"
  local python_bin
  python_bin="$(pyenv root)/versions/${version}/bin/python"
  if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: pyenv did not create $python_bin" >&2; return 1
  fi
  "$python_bin" -m venv "$project_dir/.venv"
  # shellcheck disable=SC1091
  source "$project_dir/.venv/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  echo "[SETUP] Virtual environment: $VIRTUAL_ENV"
}

setup_torch() {
  local target="${1:-gpu}"
  local index="https://download.pytorch.org/whl/cu121"
  if [[ "$target" == cpu ]]; then
    index="https://download.pytorch.org/whl/cpu"
  elif [[ "$target" != gpu ]]; then
    echo 'ERROR: first parameter must be gpu or cpu' >&2; return 2
  fi
  # Driver reporting CUDA 13.0 can run a PyTorch CUDA 12.1 wheel.
  # Poetry may have installed a same-number CPU wheel first, so check the
  # wheel build itself before deciding it is already installed.
  if TARGET_TORCH_MODE="$target" python - <<'PYCHECK'
import os, sys
try:
    import torch
    correct_version = torch.__version__.split('+')[0] == '2.4.1'
    correct_backend = ((torch.version.cuda == '12.1') if os.environ['TARGET_TORCH_MODE'] == 'gpu'
                       else (torch.version.cuda is None))
    sys.exit(0 if correct_version and correct_backend else 1)
except Exception:
    sys.exit(1)
PYCHECK
  then
    echo '[SETUP] A matching PyTorch wheel is already installed.'
  else
    python -m pip install --force-reinstall torch==2.4.1 --index-url "$index"
  fi
  python - <<'PYINNER'
import torch
print('[CHECK] PyTorch:', torch.__version__, 'built for CUDA:', torch.version.cuda)
print('[CHECK] CUDA available:', torch.cuda.is_available())
PYINNER
  if [[ "$target" == gpu ]]; then
    python - <<'PYINNER'
import torch
if not torch.cuda.is_available():
    raise SystemExit('ERROR: PyTorch cannot access NVIDIA GPU. Run nvidia-smi, see setup/TROUBLESHOOTING.md, or run setup.sh cpu')
print('[CHECK] GPU:', torch.cuda.get_device_name(0))
PYINNER
  fi
}

setup_pyg() {
  # Core MessagePassing/GAT without optional compiled PyG extensions.
  python -m pip install 'torch-geometric==2.6.1'
  python - <<'PYINNER'
import torch_geometric
from torch_geometric.nn import MessagePassing, GATConv
print('[CHECK] torch-geometric:', torch_geometric.__version__)
PYINNER
}
