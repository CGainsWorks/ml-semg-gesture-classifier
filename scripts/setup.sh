#!/usr/bin/env bash
# Set up a working environment from a clean checkout.
#
#   bash scripts/setup.sh          create .venv and install everything
#   bash scripts/setup.sh --data   also download Ninapro DB5 (about 200 MB)
#
# Python 3.10 or newer. The reported numbers were produced on 3.12.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
WITH_DATA=0
for arg in "$@"; do
    case "$arg" in
        --data) WITH_DATA=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "no '$PYTHON' on PATH. Set PYTHON=/path/to/python and try again." >&2
    exit 1
}

version="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$version" in
    3.10|3.11|3.12|3.13) ;;
    *) echo "Python $version is untested here; 3.10 to 3.13 are supported." >&2 ;;
esac

echo "creating $VENV with Python $version"
"$PYTHON" -m venv "$VENV"

if [ -x "$VENV/bin/python" ]; then
    VENV_PYTHON="$VENV/bin/python"       # Linux and macOS
else
    VENV_PYTHON="$VENV/Scripts/python"   # Windows
fi

"$VENV_PYTHON" -m pip install --upgrade pip

# The CPU torch wheel by default. The CUDA build is several gigabytes and this
# model trains in minutes on a CPU; pass TORCH_INDEX to override, for example
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash scripts/setup.sh
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
"$VENV_PYTHON" -m pip install --index-url "$TORCH_INDEX" torch==2.6.0
"$VENV_PYTHON" -m pip install -r requirements-dev.txt

if [ "$WITH_DATA" -eq 1 ]; then
    echo
    echo "downloading Ninapro DB5. Research-use licence: see data/DATASET_CARD.md"
    "$VENV_PYTHON" -m src.data.download --verify
fi

cat <<MESSAGE

Done. Activate it with:
  source $VENV/bin/activate      # Linux and macOS
  $VENV\Scripts\activate         # Windows

Then:
  make smoke   whole pipeline on synthetic data, about two minutes
  make data    download Ninapro DB5
  make demo    every number in the README, from nothing (hours)
MESSAGE
