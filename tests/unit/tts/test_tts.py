import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from unittest.mock import patch, MagicMock

def test_kokoro_tts_initialization():
    from src.tts.kokoro_tts import Kokoro_TTS
    
    # Needs to be patched since autouse fixtures cover root execution, 
    # but we are verifying internal logic if instantiated alone.
    with patch("src.tts.kokoro_tts.KModel", create=True), patch("src.tts.kokoro_tts.KPipeline", create=True):
        # Just assert it doesn't crash
        tts = Kokoro_TTS()
        assert tts is not None

def test_qwen_tts_voice_setup():
    from src.tts.qwen3_tts import QwenVoiceSetup
    
    mock_tts = MagicMock()
    mock_asr = MagicMock()
    
    with patch("os.path.exists", return_value=True): # Skip generation
        setup = QwenVoiceSetup(mock_tts, mock_asr)
        assert setup is not None

