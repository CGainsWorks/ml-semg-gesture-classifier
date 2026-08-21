#!/usr/bin/env bash
# The whole pipeline on synthetic data, in about two minutes.
#
# No download, no GPU, no Ninapro. This proves the pipeline wires together on a
# clean machine and is what CI runs.
#
# THE ACCURACY THIS PRINTS IS NOT A RESULT. The data comes from
# src/data/synthetic.py. Every artefact it writes lands under a directory
# named "synthetic" and every report it writes is stamped accordingly. For
# real numbers, run `make demo`.

set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-config/ci.yaml}"
cd "$(dirname "$0")/.."

banner() { printf '\n=== %s ===\n' "$1"; }

printf '%s\n' \
  '########################################################################' \
  '# SYNTHETIC DATA. The accuracies below are a wiring check, not results. #' \
  '########################################################################'

banner "dataset"
"$PYTHON" -m src.data.build_report --config "$CONFIG" --out artifacts/synthetic_dataset_stats.json

banner "train and cross-validate"
for model in classical_lda tensorflow pytorch; do
    "$PYTHON" -m src.train --config "$CONFIG" --model "$model" --protocol all --quiet
done

banner "deployment models"
"$PYTHON" -m src.train --config "$CONFIG" --model tensorflow --protocol deploy --quiet
"$PYTHON" -m src.train --config "$CONFIG" --model pytorch --protocol deploy --quiet

banner "quantise and export"
"$PYTHON" -m src.quantise.export --config "$CONFIG"

banner "latency"
"$PYTHON" -m src.embedded.benchmark --config "$CONFIG" --label smoke \
    --out artifacts/latency_smoke.json

banner "streaming control loop"
"$PYTHON" -m src.embedded.control_loop --config "$CONFIG" \
    --out artifacts/control_loop_smoke.json

banner "report"
"$PYTHON" -m src.evaluate --config "$CONFIG"

printf '\nPipeline runs end to end. The numbers above are synthetic. `make demo` for real ones.\n'
