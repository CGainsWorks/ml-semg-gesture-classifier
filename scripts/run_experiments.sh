#!/usr/bin/env bash
# Regenerate every number quoted in the README, from nothing.
#
# Expects the Ninapro DB5 archives under data/raw/DB5 (run `make data` first)
# and takes a few hours on a machine with a GPU. `make demo` calls this.
#
# One deliberate omission, called out here rather than hidden: the RBF SVM is
# trained under the within-subject protocol only. Its fit is quadratic in the
# training set size, and the leave-one-subject-out folds have thirteen times
# as many training windows as the within-subject ones. That is not merely an
# inconvenience: a kernel machine with tens of thousands of support vectors
# would not fit on the microcontroller this project targets either, so the
# scaling limit is a deployment fact and not just a runtime one. The README
# says so too.

set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-config/default.yaml}"
cd "$(dirname "$0")/.."

banner() { printf '\n=== %s ===\n' "$1"; }

banner "dataset statistics"
"$PYTHON" -m src.data.build_report --config "$CONFIG"

# The two ablations answer design questions rather than producing headline
# numbers, and between them they take a couple of hours. SKIP_ABLATIONS=1
# leaves their committed results alone and re-runs only the reported figures.
if [ "${SKIP_ABLATIONS:-0}" = "1" ]; then
    banner "skipping the ablations (SKIP_ABLATIONS=1); reusing their committed results"
else
    banner "architecture and augmentation ablation"
    "$PYTHON" -m experiments.architecture_comparison --config "$CONFIG"

    # Two stages: sweep thirty geometries on one fold, then re-run the best two
    # that meet the budget over all ten. The shipped geometry is chosen from
    # the second stage, not the first.
    banner "window geometry against the latency budget"
    "$PYTHON" -m experiments.latency_accuracy_tradeoff --config "$CONFIG" --confirm 2
fi

banner "classical baselines, all protocols"
for model in classical_lda classical_rf; do
    "$PYTHON" -m src.train --config "$CONFIG" --model "$model" --protocol all --quiet
done

banner "RBF SVM, within-subject only (see the note at the top of this script)"
"$PYTHON" -m src.train --config "$CONFIG" --model classical_svm \
    --protocol within_subject --quiet

banner "deep models, all protocols"
for model in tensorflow pytorch; do
    "$PYTHON" -m src.train --config "$CONFIG" --model "$model" --protocol all --quiet
done

banner "deployment models, trained on every subject"
"$PYTHON" -m src.train --config "$CONFIG" --model tensorflow --protocol deploy --quiet
"$PYTHON" -m src.train --config "$CONFIG" --model pytorch --protocol deploy --quiet

banner "quantise and export"
"$PYTHON" -m src.quantise.export --config "$CONFIG"

banner "latency on this machine"
"$PYTHON" -m src.embedded.benchmark --config "$CONFIG" --label host

banner "on-target latency estimates"
"$PYTHON" -m src.embedded.estimate_report --config "$CONFIG"

banner "streaming control loop, causal filter, int8 model"
"$PYTHON" -m src.embedded.control_loop --config "$CONFIG"

banner "figures, tables and the evaluation report"
"$PYTHON" -m src.evaluate --config "$CONFIG"

banner "signal figure for the README"
"$PYTHON" -m src.eval.make_figures --config "$CONFIG"

printf '\nEverything regenerated. Results are under analysis/, benchmarks/ and experiments/results/.\n'
