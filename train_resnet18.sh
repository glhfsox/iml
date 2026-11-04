#!/usr/bin/env bash
# Reproduce R1-R7 with the project's shared training CLI.
set -euo pipefail

case "${1:-}" in
  "") dry_run=false ;;
  --dry-run) dry_run=true ;;
  *) printf 'Usage: bash train_resnet18.sh [--dry-run]\n' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf 'Usage: bash train_resnet18.sh [--dry-run]\n' >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
data_dir="${DATA_DIR:-data/training_dataset_v2}"
topic="${TOPIC:-optimizers_reproduce}"

run_trial() {
  local label="$1" epochs="$2" optimizer="$3" lr="$4" dropout="$5" weight_decay="$6"
  local -a cmd=(
    uv run iml-train
    --config app/models/configs/resnet18.json
    --data "$data_dir" --member petr --topic "$topic"
    --set "epochs=$epochs" --set "optimizer=$optimizer"
    --set "lr=$lr" --set "dropout=$dropout" --set "weight_decay=$weight_decay"
    --set pretrained=true --set seed=42 --set batch_size=32
    --set early_stopping_patience=0
  )
  printf '\n%s\n' "$label"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$dry_run" == false ]]; then
    "${cmd[@]}"
  fi
}

# R1/R2 isolate optimizer; R2/R3 isolate dropout.
run_trial R1 10 sgd   0.0001  0.2 0.0
run_trial R2 10 adamw 0.0001  0.2 0.0
run_trial R3 10 adamw 0.0001  0.4 0.0
# R4/R5/R6 isolate weight decay; R5/R7 isolate learning rate.
run_trial R4 7  adamw 0.0001  0.1 0.0
run_trial R5 7  adamw 0.0001  0.1 0.01
run_trial R6 7  adamw 0.0001  0.1 0.05
run_trial R7 7  adamw 0.00005 0.1 0.01
