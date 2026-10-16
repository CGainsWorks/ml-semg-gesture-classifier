#!/usr/bin/env bash
# Rebuild the firmware's model and its test vectors.
#
# Trains the shipped architecture on synthetic data (config/firmware.yaml),
# quantises it to int8 TFLite with the same exporter as the main pipeline,
# copies the int8 model into firmware/model/ and writes the fixed windows and
# the desktop interpreter's outputs for them into firmware/vectors/.
#
# Only needed when the architecture or the exporter changes. The firmware
# build itself uses the committed files and needs no Python.

set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG=config/firmware.yaml
cd "$(dirname "$0")/../.."

"$PYTHON" -m src.train --config "$CONFIG" --model tensorflow --protocol deploy --quiet
"$PYTHON" -m src.quantise.export --config "$CONFIG" --no-evaluate

mkdir -p firmware/model
cp artifacts/firmware-model/deploy/synthetic/gesture_cnn_int8.tflite firmware/model/
"$PYTHON" -m src.embedded.firmware_vectors --config "$CONFIG"
