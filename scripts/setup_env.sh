#!/usr/bin/env bash
# Set up a local Python env for running the Gradio app / pipeline WITHOUT using
# the (full) home partition. Training happens on Colab — you only need this if
# you want to run the website or pipeline locally.
#
#   /home  -> 98% full (don't put venv/caches here)
#   /opt   -> 87 GB free   } root-owned: a writable dir must be created once
#   /      -> 72 GB free   }
#
# One-time, ask your admin (needs sudo) for a writable dir on a big partition:
#   sudo mkdir -p /opt/venkat && sudo chown "$USER" /opt/venkat
#
# Then:  BASE=/opt/venkat ./scripts/setup_env.sh
set -euo pipefail

BASE="${BASE:-/opt/$USER}"

if ! mkdir -p "$BASE" 2>/dev/null || [ ! -w "$BASE" ]; then
  echo "ERROR: '$BASE' is not writable."
  echo "Home is full, so a venv can't go there. Create a writable dir on a big"
  echo "partition first (one-time, needs admin):"
  echo "    sudo mkdir -p /opt/$USER && sudo chown $USER /opt/$USER"
  echo "Then re-run:  BASE=/opt/$USER $0"
  exit 1
fi

echo "Using BASE=$BASE  (free: $(df -h "$BASE" | awk 'NR==2{print $4}'))"

# Redirect ALL caches off /home
export PIP_CACHE_DIR="$BASE/.cache/pip"
export HF_HOME="$BASE/.cache/huggingface"
export TMPDIR="$BASE/tmp"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

VENV="$BASE/venvs/sarvam"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$(dirname "$0")/../app/requirements.txt"

cat <<EOF

Done. To use this env in a new shell:
    export HF_HOME="$HF_HOME" TMPDIR="$TMPDIR"
    source "$VENV/bin/activate"

Run the website locally:
    ASR_MODEL=your-username/whisper-small-ta-saaras python app/app.py
EOF
