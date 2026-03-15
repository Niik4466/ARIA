#!/bin/bash
set -e

# ==============================================================================
# ARIA & RVC INSTALLATION SCRIPT (Linux)
# ==============================================================================

GPU_MODE=""
GPU_MODEL=""
INSTALL_RVC=0

# Function to display usage
usage() {
    echo "Usage: $0 --gpu <gpu_mode> [--gpu_model <model>]"
    echo ""
    echo "Options:"
    echo "  --gpu <mode>       Specify GPU mode: 'cuda' (NVIDIA) or 'rocm' (AMD)."
    echo "  --gpu_model <model> Optional. Specify GPU model (e.g., '5090') for specific handling."
    echo "  --rvc              Optional. Install RVC backend alongside ARIA."
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
    --rvc)
        INSTALL_RVC=1
        shift 1
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
if [ "$INSTALL_RVC" -eq 1 ]; then
    echo ""
    echo "[PHASE 2] Setting up RVC Subsystem..."
    
    # Activate existing aria environment
    echo "[ARIA] Activating environment..."
    source aria_venv/bin/activate
    
    echo "[RVC] Cloning official WebUI..."
    if [ ! -d "src/rvc/rvc_webui" ]; then
        git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI src/rvc/rvc_webui
    else
        echo "[RVC] 'src/rvc/rvc_webui' already exists. Skipping clone."
    fi
    
    echo "[CRITICAL] Downgrading pip for fairseq compatibility..."
    pip install pip==23.3.1
    
    echo "[RVC] Installing RVC dependencies..."
    pip install fairseq==0.12.2 faiss-cpu ffmpeg-python praat-parselmouth pyworld torchcrepe
    
    echo "[RVC] Upgrading pip back..."
    pip install --upgrade pip
    
    echo "[RVC] Downloading base models..."
    python src/rvc/rvc_webui/tools/download_models.py
    
    echo "[ARIA] Deactivating..."
    deactivate
else
    echo ""
    echo "[INFO] Skipping RVC Installation (--rvc flag not provided)."
fi

echo ""
echo "[INSTALL] Installation Complete!"

