import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from unittest.mock import patch, MagicMock

def test_asr_transcribe():
    from src.asr import ASR
    
    with patch("src.asr.WhisperModel") as MockWhisper:
        mock_model_instance = MockWhisper.return_value
        
        class MockSegment:
            def __init__(self, text):
                self.text = text
                
        mock_model_instance.transcribe.return_value = ([MockSegment("Hello"), MockSegment("world")], None)
        
        asr = ASR()
        fake_audio = np.array([0.1, 0.2], dtype=np.float32)
        
        text = asr.speech_to_text(fake_audio)
        
        assert text == "Hello world"
        mock_model_instance.transcribe.assert_called_once()

def test_asr_transcribe_empty():
    from src.asr import ASR
    
    with patch("src.asr.WhisperModel"):
        asr = ASR()
        
        # Empty array
        text = asr.speech_to_text(np.array([], dtype=np.float32))
        assert text == ""
        
        # None
        text_none = asr.speech_to_text(None)
        assert text_none == ""
