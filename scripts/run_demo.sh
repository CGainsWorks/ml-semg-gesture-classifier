#!/usr/bin/env bash
# The one command.
#
#   bash scripts/run_demo.sh
#
# If Ninapro DB5 is on disk, this reproduces every number in the README, which
# takes a few hours. If it is not, it runs the same pipeline end to end on
# synthetic data in about two minutes and tells you how to get the real thing.
# Pass --data to download it (about 200 MB) and then run the real one.
#
# There is no third path where it half-works and quietly reports something
# meaningless: the synthetic run stamps SYNTHETIC on everything it writes.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
RAW_DIR="${RAW_DIR:-data/raw/DB5}"
FETCH=0
for arg in "$@"; do
    case "$arg" in
        --data) FETCH=1 ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ "$FETCH" -eq 1 ]; then
    echo "Fetching Ninapro DB5. Research-use licence: see data/DATASET_CARD.md"
    "$PYTHON" -m src.data.download --verify
fi

# All ten subject archives, or none of it counts as present.
present=0
for subject in 1 2 3 4 5 6 7 8 9 10; do
    [ -s "$RAW_DIR/s$subject.zip" ] && present=$((present + 1))
done

if [ "$present" -eq 10 ]; then
    echo "Ninapro DB5 found ($present/10 archives). Running the real pipeline."
    echo "This takes a few hours. Output lands in analysis/, benchmarks/ and"
    echo "experiments/results/."
    exec bash scripts/run_experiments.sh
fi

cat <<MESSAGE

Ninapro DB5 is not on disk ($present/10 archives under $RAW_DIR).

Running the pipeline on synthetic data instead. It exercises every stage in
about two minutes, and the accuracy it reports is a wiring check rather than a
result. To get the real numbers:

    bash scripts/run_demo.sh --data     download DB5 and run the real thing
    make data && make demo              the same, through the Makefile

MESSAGE

exec bash scripts/run_smoke.sh
