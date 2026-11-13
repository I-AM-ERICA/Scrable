#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required (3.11 recommended)"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[local,notebook]'
.venv/bin/python -m ipykernel install --user --name scrable --display-name 'SCRABLE (HyperCLOVA)'
echo 'Setup complete. Accept model access on Hugging Face, then run:'
echo '.venv/bin/hf auth login'
echo '.venv/bin/jupyter lab --ip=127.0.0.1 --port=8888 --no-browser --allow-root'
