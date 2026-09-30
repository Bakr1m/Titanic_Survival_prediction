#!/usr/bin/env bash
# Reproduce data/titanic_clean.csv after cloning.
# Source: seaborn's built-in Titanic dataset (Kaggle Titanic, 891 rows) —
# too duplicative to version, so data/ is gitignored.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f data/titanic_clean.csv ]; then
  echo "data/titanic_clean.csv already present — nothing to do."
  exit 0
fi

mkdir -p data
# Needs training deps (seaborn fetches the dataset): make install-train first.
.venv/bin/python -c "from src.preprocessing import prepare_data; prepare_data()"
echo "wrote data/titanic_clean.csv"
