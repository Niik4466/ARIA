# A.R.I.A - Core Components

A.R.I.A is built upon a highly modular internal structure. This ensures each specific task—speech-to-text, text-to-speech, LLM inference, memory retrieval, and wake-word detection—is compartmentalized and easily upgradable.

## 1. Agent (`src/agent.py`)
Acts as the central communication bridge with the local LLM (typically **Ollama**).
- Contains strictly formatted and engineered **System Prompts** for every situation (Decision Nodes, Tool Generation, Final Responses, Greetings, Farewells).
- Exposes helper methods like `call_ollama()` and `call_ollama_stream()` to seamlessly interface with an API endpoint, returning either full strings or asynchronous generators (streams).
- Implements text-cleaning utilities to remove markdown symbols and `<think>` tags which would otherwise ruin TTS synthesis.

## 2. RAG (`src/rag.py`)
*R*etrieval-*A*ugmented *G*eneration processes using a local vector database (**ChromaDB**).
- **Initialization**: Upon system boot, it crawls the `./documents` folder, separates text, semantic-chunks the data, and stores it by categories.
- **Inference**: Allows querying the vector database to retrieve highly relevant chunks of context based on user prompts.
- **Conversation History**: Stores the active conversation context dynamically to preserve short-to-medium-term interaction memory.

## 3. ASR (`src/asr.py`)
*A*utomatic *S*peech *R*ecognition module powered by **Faster-Whisper**.
- Continuously listens via the **VAD** (Voice Activity Detection) until a user stops speaking.
- Transcribes the incoming raw audio (NumPy array) into text using optimized CTranslate2 inference.

## 4. VAD & WakeWord (`src/vad/vad.py` & `src/vad/wakeword.py`)
Handles microphone input and environmental audio logic.
- **`VAD`**: Utilizes the Silero VAD model. It continuously evaluates the microphone stream, utilizing pre-roll buffers and sensible timeouts to neatly snip human speech from silence without arbitrarily cutting off the user.
- **`WakeWord`**: Responsible for detecting when the user calls A.R.I.A. Uses a **WavLM** feature extractor to turn audio into mathematical embeddings. Uses cosine similarity to compare ambient microphone embeddings against the known saved wake-word profiles.
- **`WakeWordSetup`**: An independent class dedicated exclusively to onboarding the user, generating the `.wav` sample templates, and using interactive TTS instructions.

## 5. TTS
*T*ext *T*o *S*peech transforms LLM text responses into audible dialogue. Modularly swapped between engines based on user configuration.
- **`Kokoro_TTS` (`src/tts/kokoro_tts.py`)**: A fast and lightweight default TTS. Capable of receiving text streams and generating grouped audio chunks asynchronously to ensure extremely quick Time-To-First-Speech (TTFS).
- **`Qwen3_TTS` (`src/tts/qwen3_tts.py`)**: An alternative engine utilized for deeper voice cloning features.
- Both modules expose the exact same interface (`generate_speech`, `generate_speech_stream`) to ensure effortless swapping inside `graph.py`.

## 6. RVC Backend (`src/tts/rvc_backend.py`)
*R*etrieval-Based *V*oice *C*onversion.
- Acts as a real-time post-processing middleware for the TTS engines.
- If enabled, A.R.I.A takes the standard robotic TTS audio and morphs its vocal timbre to mirror a high-quality model (e.g., Hatsune Miku) natively stored in the `rvc_models` directory, heavily elevating the assistant's personality footprint.

## 7. Tools (`src/Tools/`)
A dynamically loaded registry of capabilities bridging the Assistant to the real world.
- **`registry.py`**: The central master-list mapping JSON commands to Python functions.
- Categories include OS functions (reading/writing files, parsing shell commands), Basic functions (Math, Date/Time), Web Search, and **Autoconfig** (Reconfiguring WakeWords natively).

## 8. Backend Container (`src/container.py`)
Acts as the central repository and Dependency Injection framework for all core functionalities.
- Instead of maintaining chaotic global singletons scattered across `nodes.py`, `state.py`, or `main.py`, the `Container` class unifies the initialization of ASR, TTS, VAD, WakeWord, RVC, and RAG managers.
- Once created, the instance is passed securely throughout the LangGraph steps (`GraphState`), guaranteeing safe resource handling, cleaner scope visibility, and far greater testing flexibility.
