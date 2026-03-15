# --- VAD --
SAMPLE_RATE = 16000          # WebRTC VAD requires 8, 16, 32 or 48 kHz
CHANNELS = 1
VAD_MIN_SILENCE_MS = 500     # Wait before cutting off
VAD_SPEECH_PAD_MS = 300      # Buffer before/after for naturalness
VAD_AGGRESSIVENESS = 2       # 0-3 (higher = more aggressive)

# --- WakeWord ---
WAKEWORD_SAMPLES = 5         # Number of samples required for wakeword registration
WAKEWORD_THRESHOLD = 0.90    # Cosine similarity threshold for detection

# --- STT ---
ASR_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
ASR_LANGUAGE = "en"
ASR_DEVICE = "cuda"         # "cuda" or "cpu"
ASR_COMPUTE_TYPE = "int8"    # "float16" for powerful GPU, "int8" for balanced
ASR_BEAM_SIZE = 5

# --- Agent (Ollama by default) ---
USER_NAME = "Nik"        # User Name
API_URL = "http://127.0.0.1:11434/api/generate"
API_TYPE = "ollama"
RESPONSE_MODEL = "qwen3.5:4b"
DECISOR_MODEL = "qwen3.5:4b"

AGENT_EXTRA_PROMPT= """ 
You have a calm, neutral, and polite personality. You communicate clearly and respectfully.
"""

# --- TTS ---
USE_QWEN3_TTS = True      # True = use Qwen3-TTS, False = use Kokoro-tts
QWEN3_LANG = "English"
FLASH_ATTENTION = False   # True = Requires flash_attn library installed, False = Disabled
KOKORO_LANG = "a"          # Language code for KPipeline (e.g., 'a', 'e', 'p', etc.)
KOKORO_VOICE = "af_heart"        # Kokoro voice name (e.g., 'af_heart', 'af_sarah', etc.)
KOKORO_SPEED = 1.0       # Speed factor (1.0 = normal, >1 faster, <1 slower)

# --- RVC TTS ---
RVC_MODEL = "rvc_voice"  # Name of your model folder inside rvc_models/
USE_RVC = False          # Change to False to use base TTS only

# --- RAG ---
DOCUMENTS_PATH = "./documents"

# --- Debug ---
verbose_mode = True