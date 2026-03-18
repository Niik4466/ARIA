"""
Container Module.

Purpose: Centralize the initialization and lifecycle of the core ARIA backend
components to avoid scattered global state.
"""

import os
import sys

# Ensure ARIA root is in PATH for all imports
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

from src.utils import Config
from src.vad.vad import VAD
from src.vad.wakeword import WakeWord, WakeWordSetup
from src.asr import ASR
from src.rag import rag_manager
from src.audio_io import player

class Container:
    """
    Dependency Injection Container for ARIA backend.
    Holds all necessary global models and configurations.
    """
    def __init__(self):
        print("\n--- INITIALIZING BACKEND CONTAINER ---")
        self.config = Config()
        self.use_qwen3_tts = self.config.get("USE_QWEN3_TTS")
        self.use_rvc = self.config.get("USE_RVC")
        
        # Initialize Core Components
        self.vad = VAD()
        self.asr = ASR(self.vad)
        
        if self.use_qwen3_tts:
            from src.tts.qwen3_tts import Qwen3_TTS, QwenVoiceSetup
            self.tts = Qwen3_TTS()
            self.qwen_voice_setup = QwenVoiceSetup(self.tts, self.asr)
        else:
            from src.tts.kokoro_tts import Kokoro_TTS
            self.tts = Kokoro_TTS()
            
        if self.use_rvc:
            from src.tts.rvc_backend import RVC_Backend
            self.rvc = RVC_Backend()
        else:
            self.rvc = None
            
        self.audio_player = player
        
        # Wakeword and Samples
        self.samples_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "vad", "samples")
        )
        
        self.ww_setup = WakeWordSetup(
            self.vad, 
            self.samples_dir, 
            tts=self.tts, 
            rvc=self.rvc, 
            audio_player=self.audio_player
        )
        if not self.ww_setup.has_enough_samples():
            self.ww_setup.new_wakeword_samples()
            
        self.wake_word = WakeWord(self.vad, samples_dir=self.samples_dir)
        
        # RAG Initialization
        print("🔄 Initializing RAG system...")
        self.rag_manager = rag_manager
        self.rag_categories = self.rag_manager.update()
        
        # Construct descriptions string
        desc_list = [f"- '{k}': {v}" for k, v in self.rag_categories.items()]
        self.rag_categories_desc_str = "\n".join(desc_list) if desc_list else "(No documents available)"
        print("✅ RAG Initialized.")
        print("--- BACKEND CONTAINER READY ---\n")
