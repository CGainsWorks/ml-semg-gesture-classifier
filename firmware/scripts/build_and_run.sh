#!/usr/bin/env bash
# Cross-build the firmware and run it under Renode, then compare its outputs
# with the desktop interpreter's and write the emulator report.
#
# Runs inside the image from docker/firmware.Dockerfile, which carries the Arm
# GNU Toolchain, CMake, the TFLite Micro tree and Renode:
#
#   docker build -f docker/firmware.Dockerfile -t semg-firmware docker
#   docker run --rm -v "$PWD:/work" -w /work semg-firmware bash firmware/scripts/build_and_run.sh
#
# Exit status is non-zero if the build fails, if the image does not fit the
# STM32F411 memory map, if the firmware does not finish, or if any output
# differs from the reference.

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$PWD"
BUILD="${BUILD:-build/firmware}"
OUT="${OUT:-$BUILD/renode_cortex_m4.json}"

cmake -S firmware -B "$BUILD" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/firmware/cmake/arm-none-eabi.cmake" \
    -DTFLM_TREE="${TFLM_TREE:-/opt/tflm}"
cmake --build "$BUILD"

rm -f "$BUILD/uart.log"
cat > "$BUILD/run.resc" <<EOF
\$repl = @$ROOT/firmware/renode/stm32f411.repl
\$elf = @$ROOT/$BUILD/semg_firmware.elf
\$uart = @$ROOT/$BUILD/uart.log
include @$ROOT/firmware/renode/run.resc
EOF
renode --disable-xwt --console --plain "$BUILD/run.resc" | tee "$BUILD/renode.log"

python3 firmware/tools/renode_report.py \
    --uart "$BUILD/uart.log" \
    --size "$BUILD/semg_firmware.size.txt" \
    --expected firmware/vectors/expected.json \
    --renode-version "$(renode --version 2>/dev/null | head -n 1)" \
    --gcc-version "$(arm-none-eabi-gcc --version | head -n 1)" \
    --cmake-version "$(cmake --version | head -n 1)" \
    --tflm-commit "$(cat "${TFLM_TREE:-/opt/tflm}/TFLM_COMMIT")" \
    --out "$OUT"
