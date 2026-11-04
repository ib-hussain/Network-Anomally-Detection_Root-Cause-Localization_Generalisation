#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [[ $# -ne 1 ]] || [[ "$1" != swat && "$1" != telco && "$1" != ute ]]; then
  echo 'Usage: bash train.sh swat|telco|ute' >&2
  exit 2
fi
DATASET="$1"
mkdir -p logs_train
for model in gru gcn gdn mtad_gat; do
  config="models/${model}/params_${DATASET}.yaml"
  if [[ ! -f "$config" ]]; then echo "Missing configuration $config" >&2; exit 2; fi
  echo "Training $model on $DATASET (see logs_train/${model}_${DATASET}.log)"
  if ! python models/train.py --dataset "$DATASET" --model "$model" --params_file "$config" > "logs_train/${model}_${DATASET}.log" 2>&1; then
    echo "FAILED: $model. Inspect logs_train/${model}_${DATASET}.log" >&2
    exit 1
  fi
done
