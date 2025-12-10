# --- Audio ---
SAMPLE_RATE = 16000          # WebRTC VAD exige 8, 16, 32 o 48 kHz
CHANNELS = 1
FRAME_MS = 20                # 10, 20 o 30 ms para VAD
VAD_AGGRESSIVENESS = 2       # 0-3 (más alto = más agresivo)

# --- STT ---
FASTER_WHISPER_MODEL = "small"   # "tiny", "base", "small", "medium", "large-v3" etc.
FASTER_WHISPER_DEVICE = "cpu"   # "cuda" si tienes GPU

# --- Agente (Ollama por defecto) ---
API_URL = "http://127.0.0.1:11434/api/generate"
API_TYPE = "ollama"
RESPONSE_MODEL = "qwen3"
DECISOR_MODEL = "qwen3:0.6b"

AGENT_EXTRA_PROMPT= """ 
Eres un asistente de voz con la personalidad de la vocaloid Miku, eres alegre y divertida, siempre dispuesta a ayudar al usuario.
"""

# --- TTS ---
VOICE_NAME = "es-CL-LorenzoNeural" 

# --- RVC TTS ---
RVC_API_URL = "http://localhost:5050"
RVC_MODEL = "miku_default"  # Cambia al nombre de tu modelo RVC cargado
USE_RVC = True          # Cambia a False para usar solo tts en su lugar

# --- Kokoro TTS ---
USE_KOKORO = True          # True = usar KokoroTTS en lugar de edge-tts
KOKORO_LANG = "e"          # Código de idioma para KPipeline (p.ej. 'a', 'e', 'p', etc.)
KOKORO_VOICE = "af_sarah"        # Nombre de la voz de Kokoro (p.ej. 'af_heart', 'af_sarah', etc.)
KOKORO_SPEED = 0.94       # Factor de velocidad (1.0 = normal, >1 más rápido, <1 más lento)

# RAG
DOCUMENTS_PATH = "./documents"