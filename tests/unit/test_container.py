import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from unittest.mock import patch

def test_container_initialization(clean_config):
    """
    Tests that the container initializes its properties correctly
    without crashing due to missing dependencies (thanks to autouse mocks).
    """
    from src.container import Container
    
    # We must mock config specifically so we can toggle usage parameters efficiently
    with patch("src.utils.Config.get") as mock_get, \
         patch("src.container.ASR"), \
         patch("src.container.RAGManager"), \
         patch("src.container.MCPManager"), \
         patch("src.container.MemoryManager"):
        
        # Mock answers for initial values
        def mock_config_get(key, default=None):
            if key == "USE_QWEN3_TTS": return False
            if key == "USE_RVC": return False
            return "default"
        mock_get.side_effect = mock_config_get
        
        container = Container()
        
        assert container.asr is not None
        assert getattr(container, "tts", None) is not None
        assert container.rvc is None # Since we dictated USE_RVC=False
        assert container.rag_manager is not None
        assert container.mcp_manager is not None

def test_container_loads_qwen3_tts_when_configured(clean_config):
    from src.container import Container
    
    with patch("src.utils.Config.get") as mock_get, \
         patch("src.container.ASR"), \
         patch("src.container.RAGManager"), \
         patch("src.container.MCPManager"), \
         patch("src.container.MemoryManager"), \
         patch("src.tts.qwen3_tts.Qwen3_TTS"), \
         patch("src.tts.rvc_backend.RVC_Backend"):
        
        def mock_config_get(key, default=None):
            if key == "USE_QWEN3_TTS": return True
            if key == "USE_RVC": return True
            return "default"
        mock_get.side_effect = mock_config_get
        
        container = Container()
        
        assert getattr(container, "qwen_voice_setup", None) is not None
        assert container.rvc is not None
