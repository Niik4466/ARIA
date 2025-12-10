#!/bin/bash
set -e

# ==============================================================================
# ARIA & RVC INSTALLATION SCRIPT (Linux)
# ==============================================================================

GPU_MODE=""
GPU_MODEL=""

# Function to display usage
usage() {
    echo "Usage: $0 --gpu <gpu_mode> [--gpu_model <model>]"
    echo ""
    echo "Options:"
    echo "  --gpu <mode>       Specify GPU mode: 'cuda' (NVIDIA) or 'rocm' (AMD)."
    echo "  --gpu_model <model> Optional. Specify GPU model (e.g., '5090') for specific handling."
    echo "  --help             Show this help message."
    echo ""
    exit 1
}


# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      usage
      ;;
    --gpu)
      GPU_MODE="$2"
      shift 2
      ;;
    --gpu_model)
      GPU_MODEL="$2"
      shift 2
      ;;
    *)
      shift # Unknown option
      ;;
  esac
done


if [ -z "$GPU_MODE" ]; then
    echo "[ERROR] --gpu argument is required!"
    usage
fi

echo "[INSTALL] Starting Installation Sequence..."
echo "[INFO] GPU_MODE: $GPU_MODE"
echo "[INFO] GPU_MODEL: $GPU_MODEL"


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
python -m pip install --upgrade pip
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

echo "[RVC] Configuring PyTorch for $GPU_MODE..."

INSTALLED_TORCH=0

if [[ "$GPU_MODE" == "rocm" ]]; then
    echo "[TORCH] Installing ROCm 6.4 compatible torch..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
    INSTALLED_TORCH=1
fi

# Fallback / Default
if [ "$INSTALLED_TORCH" -eq 0 ]; then
    # Note: Logic for RTX 50 series was specified for Windows only in the prompt.
    echo "[TORCH] Installing standard torch versions..."
    pip install torch torchvision
fi

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
