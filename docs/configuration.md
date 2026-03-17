# Configuration Tutorial (`config.json`)

A.R.I.A is designed to be highly customizable. By tweaking the parameters within `config.json` at the root of the project, you can dictate resource allocation, performance, and behavior.

Below is an overview of the key configurable domains and the available options for each variable.

## 1. VAD (Voice Activity Detection) Environment
Controls how A.R.I.A interprets silence and microphone events.
```json
"SAMPLE_RATE": 16000,
"CHANNELS": 1,
"VAD_MIN_SILENCE_MS": 500,
"VAD_SPEECH_PAD_MS": 300,
"VAD_AGGRESSIVENESS": 2
```
- **`SAMPLE_RATE`**: The audio capture rate. WebRTC VAD requires `8000`, `16000`, `32000`, or `48000` (Hz).
- **`CHANNELS`**: Number of audio channels. Should generally be `1` (mono).
- **`VAD_MIN_SILENCE_MS`**: Wait duration (in milliseconds) before assuming the user is done speaking. *Tip: Increase if you pause to think frequently while speaking to prevent A.R.I.A from interrupting you prematurely.*
- **`VAD_SPEECH_PAD_MS`**: Audio padding (in milliseconds) attached before and after the speech payload for naturalness.
- **`VAD_AGGRESSIVENESS`**: Scale from `0` to `3` (higher means more aggressive filtering and strips more noise).

## 2. WakeWord Configuration
```json
"WAKEWORD_SAMPLES": 5,
"WAKEWORD_THRESHOLD": 0.90
```
- **`WAKEWORD_SAMPLES`**: Total activation-word samples requested upon initial configuration.
- **`WAKEWORD_THRESHOLD`**: Cosine similarity threshold for triggering the wakeword. *Tip: If the assistant triggers accidentally from background noise, increase this slightly (e.g., to 0.92).*

## 3. ASR (Automatic Speech Recognition)
Controls the Faster-Whisper transcription model logic.
```json
"ASR_MODEL_ID": "deepdml/faster-whisper-large-v3-turbo-ct2",
"ASR_LANGUAGE": "en",
"ASR_DEVICE": "cuda",
"ASR_COMPUTE_TYPE": "int8",
"ASR_BEAM_SIZE": 5
```
- **`ASR_MODEL_ID`**: The Faster-Whisper HuggingFace model ID to be downloaded.
- **`ASR_LANGUAGE`**: Standard language assumption (e.g., `"en"` for English, `"es"` for Spanish).
- **`ASR_DEVICE`**: Use `"cuda"` for GPU acceleration or `"cpu"` for software execution.
- **`ASR_COMPUTE_TYPE`**: Precision type. `"float16"` (requires more VRAM but faster) or `"int8"` (lighter memory footprint).
- **`ASR_BEAM_SIZE`**: The beam search size for transcription. `5` is a good balance between speed and accuracy.

## 4. LLM Backend (Agent logic)
Instructs A.R.I.A where to reach its intelligence core. By default, it hits a local Ollama server instance.
```json
"API_URL": "http://127.0.0.1:11434/api/generate",
"API_TYPE": "ollama",
"RESPONSE_MODEL": "qwen3.5:4b",
"DECISOR_MODEL": "qwen3.5:4b",
"USER_NAME": "User",
"AGENT_EXTRA_PROMPT": "\nYou have a calm, neutral, and polite personality. You communicate clearly and respectfully.\n"
```
- **`API_URL`**: The endpoint for generation.
- **`API_TYPE`**: The type of backend API (default `"ollama"`).
- **`RESPONSE_MODEL`**: The main LLM for answering questions and generating responses.
- **`DECISOR_MODEL`**: The LLM used for routing and decision-making. Can be mapped to a smaller model for faster routing operations.
- **`USER_NAME`**: The name of the user interacting with A.R.I.A.
- **`AGENT_EXTRA_PROMPT`**: Additional system prompt text injected into A.R.I.A.'s context to define its personality.

## 5. Text-To-Speech (TTS) Engines
Determines what raw engine generates the voice. 
```json
"USE_QWEN3_TTS": true,
"QWEN3_LANG": "English",
"FLASH_ATTENTION": false,
"KOKORO_LANG": "a",
"KOKORO_VOICE": "af_heart",
"KOKORO_SPEED": 1.0
```
- **`USE_QWEN3_TTS`**: Set `true` to switch entirely to Qwen3-TTS, or `false` to use Kokoro TTS.
- **`QWEN3_LANG`**: Language target for Qwen3 TTS configuration (e.g., `"English"`, `"Spanish"`).
- **`FLASH_ATTENTION`**: Set `true` if you have the `flash_attn` library installed for faster performance, `false` to disable.
- **`KOKORO_LANG`**: Kokoro Pipeline language code (e.g., `"a"` for American English, `"b"` for British, `"e"` for Spanish, `"p"` for Portuguese).
- **`KOKORO_VOICE`**: Kokoro voice identifier (e.g., `"af_heart"`, `"af_sarah"`).
- **`KOKORO_SPEED`**: Talk speed factor (`1.0` = normal, `>1` = faster, `<1` = slower).

## 6. RVC (Retrieval-based Voice Conversion)
Handles the morphing wrapper on top of the generic TTS. 
```json
"USE_RVC": false,
"RVC_MODEL": "rvc_voice"
```
- **`USE_RVC`**: Change to `true` if you want to use RVC to morph the voice, `false` otherwise.
- **`RVC_MODEL`**: Expected directory name of the model inside the `/rvc_models/` folder.
*Note: Using RVC drastically improves vocal quality and tone, but demands marginally more compute resources overhead.*

## 7. RAG Documents
```json
"DOCUMENTS_PATH": "./documents"
```
- **`DOCUMENTS_PATH`**: The relative or absolute path to the knowledge base.
*Tip: A.R.I.A scans this specified path. Dropping `.txt` or `.md` files here guarantees that A.R.I.A will digest them into her local context engine on the next boot.*

## 8. Debugging
```json
"verbose_mode": true
```
- **`verbose_mode`**: `true` or `false` to toggle terminal debug spam.
