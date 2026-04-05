import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from unittest.mock import patch, MagicMock

def test_asr_listen_success():
    from src.asr import ASR
    mock_vad = MagicMock()
    
    with patch("src.asr.WhisperModel") as MockWhisper, patch("src.asr.VADAudioStream") as MockStream:
        mock_stream_instance = MockStream.return_value
        fake_audio = np.array([0.1, 0.2], dtype=np.float32)
        mock_stream_instance.get_next_segment.return_value = fake_audio
        
        asr = ASR(mock_vad)
        
        result = asr.listen(timeout=1.0)
        
        asr.stream.start.assert_called_once()
        asr.stream.stop.assert_called_once()
        assert np.array_equal(result, fake_audio)

def test_asr_transcribe():
    from src.asr import ASR
    mock_vad = MagicMock()
    
    with patch("src.asr.WhisperModel") as MockWhisper, patch("src.asr.VADAudioStream"):
        mock_model_instance = MockWhisper.return_value
        
        class MockSegment:
            def __init__(self, text):
                self.text = text
                
        mock_model_instance.transcribe.return_value = ([MockSegment("Hello"), MockSegment("world")], None)
        
        asr = ASR(mock_vad)
        fake_audio = np.array([0.1, 0.2], dtype=np.float32)
        
        text = asr.speech_to_text(fake_audio)
        
        assert text == "Hello world"
        mock_model_instance.transcribe.assert_called_once()

def test_asr_transcribe_empty():
    from src.asr import ASR
    mock_vad = MagicMock()
    
    with patch("src.asr.WhisperModel"), patch("src.asr.VADAudioStream"):
        asr = ASR(mock_vad)
        
        # Empty array
        text = asr.speech_to_text(np.array([], dtype=np.float32))
        assert text == ""
        
        # None
        text_none = asr.speech_to_text(None)
        assert text_none == ""
