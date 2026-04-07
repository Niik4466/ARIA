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
from src.asr import ASR
from src.rag import RAGManager
from src.mcp.mcp_manager import MCPManager
from src.memory import MemoryManager
from src.logger import mlog as print

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
        self.asr = ASR()
        
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
        
        # RAG Initialization
        print("🔄 Initializing RAG system...")
        self.rag_manager = RAGManager()
        self.rag_categories = self.rag_manager.update()
        
        # Construct descriptions string
        desc_list = [f"- '{k}': {v}" for k, v in self.rag_categories.items()]
        self.rag_categories_desc_str = "\n".join(desc_list) if desc_list else "(No documents available)"
        print("✅ RAG Initialized.")

        # MCP Initialization
        print("🔄 Initializing MCP system...")
        self.mcp_manager = MCPManager(self.rag_manager)
        print("✅ MCP Initialized.")

        # Memory Initialization
        print("🔄 Initializing Memory system...")
        self.memory_manager = MemoryManager(self.rag_manager)
        print("✅ Memory Initialized.")

        print("--- BACKEND CONTAINER READY ---\n")
