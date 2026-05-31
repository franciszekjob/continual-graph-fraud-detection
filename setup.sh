#!/bin/bash
set -euo pipefail

# ── 1. uv ──────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── 2. Python 3.12 ─────────────────────────────────────────────────────────
if ! uv python list --only-installed 2>/dev/null | grep -q "3\.12"; then
    echo "Installing Python 3.12..."
    uv python install 3.12
fi

# ── 3. venv + dependencies ─────────────────────────────────────────────────
echo "Syncing dependencies..."
uv sync

# ── 4. directories ─────────────────────────────────────────────────────────
mkdir -p data/raw results

# ── 5. kaggle credentials ──────────────────────────────────────────────────
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [ ! -f "$KAGGLE_JSON" ] && [ -f "kaggle.json" ]; then
    echo "Kopiuję kaggle.json do ~/.kaggle/..."
    mkdir -p "$HOME/.kaggle"
    cp kaggle.json "$KAGGLE_JSON"
    chmod 600 "$KAGGLE_JSON"
fi

# ── 6. dataset ─────────────────────────────────────────────────────────────
if [ -f "data/raw/train_transaction.csv" ]; then
    echo "Dataset already present, skipping download."
elif [ -f "$KAGGLE_JSON" ]; then
    echo "Downloading IEEE-CIS Fraud Detection dataset..."
    uv run kaggle competitions download -c ieee-fraud-detection -p data/raw/
    echo "Unzipping..."
    unzip -q data/raw/ieee-fraud-detection.zip -d data/raw/
    rm data/raw/ieee-fraud-detection.zip
else
    echo ""
    echo "INFO: Brak ~/.kaggle/kaggle.json. Pobierz z kaggle.com/settings → API → Create New Token"
    echo "  Wrzuć kaggle.json do ~/.kaggle/ lub do katalogu projektu — skrypt skopiuje go automatycznie."
fi

echo ""
echo "Setup complete. Run experiment:"
echo "  uv run python experiments/run_experiment.py"
