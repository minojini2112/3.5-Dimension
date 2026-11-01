#!/usr/bin/env bash
# One-shot setup for Linux/macOS. Usage: bash setup.sh [cpu|cu121]
set -e
FLAVOR=${1:-cpu}
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/$FLAVOR
pip install -r requirements.txt
python download_weights.py   # optional: app.py / run.py also do this on first start
echo "Done. Run: source .venv/bin/activate && python app.py"
