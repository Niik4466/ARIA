#!/bin/bash
set -e

# ==============================================================================
# ARIA SYSTEM INITIALIZATION SCRIPT
# ==============================================================================

# 1. Start Ollama Service
echo "[INIT] Checking Ollama service..."
if ! command -v ollama &> /dev/null; then
    echo "[ERROR] 'ollama' command not found. Please install Ollama."
    exit 1
fi

if ! pgrep -x "ollama" > /dev/null; then
    echo "[INIT] Starting Ollama serve..."
    ollama serve &
    # Allow some time for startup
    sleep 5
else
    echo "[INIT] Ollama is already running."
fi

# Wait for Ollama API to be responsive
echo "[INIT] Waiting for Ollama API to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done
echo "[INIT] Ollama is ready."

# 2. Pull Required Models
echo "[INIT] Pulling required models..."

# Extract models from config.py
RESPONSE_MODEL=$(python3 -c "import config; print(config.RESPONSE_MODEL)")
DECISOR_MODEL=$(python3 -c "import config; print(config.DECISOR_MODEL)")

echo "[INFO] Pulling RESPONSE_MODEL: $RESPONSE_MODEL"
ollama pull "$RESPONSE_MODEL"

if [ "$RESPONSE_MODEL" != "$DECISOR_MODEL" ]; then
    echo "[INFO] Pulling DECISOR_MODEL: $DECISOR_MODEL"
    ollama pull "$DECISOR_MODEL"
fi

echo "[INFO] Pulling embedding model: nomic-embed-text"
ollama pull nomic-embed-text

# 3. Start RVC API in background
echo "[INIT] Starting RVC API..."
if [ ! -d "rvc/rvc_venv" ]; then
    echo "[ERROR] 'rvc/rvc_venv' not found. Please run install-linux.sh first."
    exit 1
fi

# Create logs directory
if [ ! -d "logs" ]; then
    mkdir -p logs
fi

# Using a subshell to switch context and run in background attached to this session
(
    cd rvc
    source rvc_venv/bin/activate
    echo "[RVC] Launching API on port 5050 (device: cuda)..."
    # Run in background with & (bound to session), silent output
    python -m rvc_python api -p 5050 -l --device cuda > ../logs/rvc_api.log 2>&1 &
)

# Initial wait for RVC
sleep 5

# 4. Start ARIA Main Application
echo "[INIT] Starting ARIA Main Application..."
if [ ! -d "aria_venv" ]; then
    echo "[ERROR] 'aria_venv' not found. Please run install-linux.sh first."
    exit 1
fi

source aria_venv/bin/activate
echo "[ARIA] Running main.py..."
python main.py
