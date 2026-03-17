"""
Processing Graph State Module.

Purpose: Define the GraphState type definition and initialize global backend instances
(ASR, TTS, VAD, RVC, Wakeword, RAG) for the voice processing pipeline.
"""

from typing import TypedDict
import os
import sys

# Ensure ARIA root is in PATH for all imports
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)
from ..utils import Config
from src.vad.vad import VAD
from src.vad.wakeword import WakeWord, WakeWordSetup
from src.asr import ASR

config = Config()

USE_QWEN3_TTS = config.get("USE_QWEN3_TTS")
USE_RVC = config.get("USE_RVC")

if USE_QWEN3_TTS:
    from src.tts.qwen3_tts import Qwen3_TTS as Qwen3TTS
    from src.tts.qwen3_tts import QwenVoiceSetup
else:
    from src.tts.kokoro_tts import Kokoro_TTS as KokoroTTS

from src.tts.rvc_backend import RVC_Backend
from src.rag import rag_manager
from src.audio_io import player

# --- RAG Initialization ---
print("🔄 Initializing RAG system...")
RAG_CATEGORIES = rag_manager.update()
RAG_CATEGORIES_DESC_STR = "\n".join([f"- '{k}': {v}" for k, v in RAG_CATEGORIES.items()])
if not RAG_CATEGORIES_DESC_STR:
    RAG_CATEGORIES_DESC_STR = "(No documents available)"
print("✅ RAG Initialized.")

# --- Graph State ---
class GraphState(TypedDict, total=False):
    # Transcribed text from user
    user_text: str
    
    # State context
    history_context: str       # Accumulated chat text
    tools_context: str         # Accumulated text with tool results
    iteration_count: int       # Counter for loop protection
    
    # RAG results
    rag_category: str          # Selected RAG category
    rag_context: str           # Retrieved text

    # Flow control
    next_node: str             # 'tool_node', 'response_node' or 'end'
    selected_category: str     # 'search', 'os', 'basic', etc.
    
    # Final response
    reply_text: str
    
    # Metrics
    start_time: float


# --- Global Instances ---
vad = VAD()
asr = ASR(vad)
if USE_QWEN3_TTS:
    tts = Qwen3TTS()
    qwenVoiceSetup = QwenVoiceSetup(tts, asr)
else:
    tts = KokoroTTS()

rvc = None
if USE_RVC:
    rvc = RVC_Backend()

audio_player = player

samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "vad", "samples"))

# Setup Wakeword if necessary
ww_setup = WakeWordSetup(vad, samples_dir, tts=tts, rvc=rvc, audio_player=audio_player)
if not ww_setup.has_enough_samples():
    ww_setup.new_wakeword_samples()

# Instantiate the listening wakeword
wake_word = WakeWord(vad, samples_dir=samples_dir)
