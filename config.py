from pathlib import Path

# --- Audio ---
SAMPLE_RATE = 16000          # WebRTC VAD exige 8, 16, 32 o 48 kHz; usamos 16 kHz
CHANNELS = 1
FRAME_MS = 20                # 10, 20 o 30 ms para VAD
VAD_AGGRESSIVENESS = 2       # 0-3 (más alto = más agresivo)

# --- STT ---
FASTER_WHISPER_MODEL = "medium"   # "tiny", "base", "small", "medium", "large-v3" etc.
FASTER_WHISPER_DEVICE = "cpu"   # "cuda" si tienes GPU

# --- Agente (Ollama por defecto) ---
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen3"    # el que tengas disponible localmente

# --- TTS ---
VOICE_NAME = "es-ES-AlvaroNeural"  # si usas edge-tts; cambia si usas Kokoro
