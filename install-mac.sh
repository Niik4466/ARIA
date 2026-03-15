#!/bin/bash
set -e

# ==============================================================================
# ARIA & RVC INSTALLATION SCRIPT (macOS)
# ==============================================================================

echo "[INSTALL] Starting Installation Sequence (macOS)..."

INSTALL_RVC=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --rvc)
        INSTALL_RVC=1
        shift 1
        ;;
    --help)
        echo "Usage: $0 [--rvc]"
        echo "  --rvc    Install RVC backend alongside ARIA."
        exit 1
        ;;
    *)
        shift # Unknown option
        ;;
    esac
done

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
    
    echo "[RVC] Configuring PyTorch for macOS..."
    pip install torch torchvision
    
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
