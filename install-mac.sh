#!/bin/bash
set -e

# ==============================================================================
# ARIA & RVC INSTALLATION SCRIPT (macOS)
# ==============================================================================

echo "[INSTALL] Starting Installation Sequence (macOS)..."

# ==============================================================================
# PHASE 1: ARIA (Main System)
# ==============================================================================
echo ""
echo "[PHASE 1] Setting up ARIA environment..."

if [ ! -d "aria_venv" ]; then
    echo "[ARIA] Creating venv 'aria_venv'..."
    python3 -m venv aria_venv
else
    echo "[ARIA] 'aria_venv' already exists."
fi

echo "[ARIA] Activating environment..."
source aria_venv/bin/activate

echo "[ARIA] Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt not found in root!"
fi

echo "[ARIA] Deactivating..."
deactivate

# ==============================================================================
# PHASE 2: RVC (Subsystem)
# ==============================================================================
echo ""
echo "[PHASE 2] Setting up RVC subsystem..."

if [ ! -d "rvc" ]; then
    echo "[ERROR] 'rvc' directory not found!"
    exit 1
fi

cd rvc

if [ ! -d "rvc_venv" ]; then
    echo "[RVC] Creating venv 'rvc_venv' using Python 3.10..."
    # Attempt to find python3.10 specifically, often installed via brew
    if command -v python3.10 &> /dev/null; then
        python3.10 -m venv rvc_venv
    else
        echo "[WARNING] python3.10 not found. Configuring with default python3..."
        python3 -m venv rvc_venv
    fi
else
    echo "[RVC] 'rvc_venv' already exists."
fi

echo "[RVC] Activating environment..."
source rvc_venv/bin/activate

echo "[CRITICAL] Downgrading pip to 23.0.1..."
python3 -m pip install pip==23.0.1

echo "[RVC] Installing rvc-python..."
pip install rvc-python tensorboardX

echo "[RVC] Configuring PyTorch for macOS..."
# Default installation as requested for Mac (MPS support is included in standard builds usually, or CPU fallback)
pip3 install torch torchvision

# ==============================================================================
# PHASE 3: HOTFIX
# ==============================================================================
echo ""
echo "[PHASE 3] Applying RVC Code Hotfix..."
python3 fix_rvc_code.py

echo ""
echo "[INSTALL] Installation Complete!"
deactivate
cd ..
