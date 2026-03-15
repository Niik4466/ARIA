# Configuration Tutorial (`config.py`)

A.R.I.A is designed to be highly customizable. By tweaking the parameters within `config.py` at the root of the project, you can dictate resource allocation, performance, and behavior.

Below is an overview of the key configurable domains:

## 1. VAD (Voice Activity Detection) Environment
Controls how A.R.I.A interprets silence and microphone events.
```python
SAMPLE_RATE = 16000          # WebRTC VAD requires 8, 16, 32 or 48 kHz
CHANNELS = 1
VAD_MIN_SILENCE_MS = 500     # Wait duration before assuming user is done speaking
VAD_SPEECH_PAD_MS = 300      # Audio padding attached before/after the speech payload
VAD_AGGRESSIVENESS = 2       # Scale from 0 to 3 (higher strips more noise)
```
*Tip: Increase `VAD_MIN_SILENCE_MS` if you pause to think frequently while speaking to prevent A.R.I.A from interrupting you prematurely.*

## 2. WakeWord Configuration
```python
WAKEWORD_SAMPLES = 5         # Total activation-word samples requested upon configuration
WAKEWORD_THRESHOLD = 0.90    # Cosine similarity threshold for triggering
```
*Tip: If the assistant triggers accidentally from background noise, increase the `WAKEWORD_THRESHOLD` slightly (e.g., to 0.92).*

## 3. ASR (Automatic Speech Recognition)
Controls the Faster-Whisper transcription model logic.
```python
ASR_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
ASR_LANGUAGE = "es"          # Standard language assumption
ASR_DEVICE = "cuda"          # "cuda" for GPU acceleration or "cpu"
ASR_COMPUTE_TYPE = "int8"    # "float16" (More VRAM required) or "int8" (Lighter)
ASR_BEAM_SIZE = 5
```

## 4. LLM Backend (Agent logic)
Instructs A.R.I.A where to reach its intelligence core. By default, it hits a local Ollama server instance.
```python
API_URL = "http://127.0.0.1:11434/api/generate"
API_TYPE = "ollama"
RESPONSE_MODEL = "qwen3.5:4b"
DECISOR_MODEL = "qwen3.5:4b"  # Can be mapped to a smaller model for faster routing 

AGENT_EXTRA_PROMPT= """ 
You are a voice assistant, serius and direct.
"""

USER_NAME = "User"        # User Name
```



## 5. Text-To-Speech (TTS) Engines
Determines what raw engine generates the voice. 
```python
USE_QWEN3_TTS = False        # Set True to switch entirely to Qwen3-TTS
QWEN3_LANG = "Spanish"       # Language target for Qwen3 TTS configuration
KOKORO_LANG = "e"            # Kokoro Pipeline language code
KOKORO_VOICE = "jf_alpha"    # Kokoro Voice identifier
KOKORO_SPEED = 1.0           # Talk Speed Factor
```

## 6. RVC (Retrieval-based Voice Conversion)
Handles the morphing wrapper on top of the generic TTS. 
```python
USE_RVC = True                 # Change to False if you don't care to morph voices
RVC_MODEL = "HatsuneMiku"      # Expected directory name inside /rvc_models/
```
*Note: Using RVC drastically improves vocal quality and tone, but demands marginally more compute resources overhead.*

## 7. RAG Documents
```python
DOCUMENTS_PATH = "./documents" 
```
*Tip: A.R.I.A scans this specified path. Dropping `.txt` or `.md` files here guarantees that A.R.I.A will digest them into her local context engine on the next boot.*

## 8. Debugging
```python
verbose_mode = False         # Toggle terminal debug spam
```
