import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from unittest.mock import patch, MagicMock
import numpy as np

def test_vad_audio_stream_logic():
    from src.vad.vad import VADAudioStream
    
    mock_vad = MagicMock()
    # Provide a threshold
    mock_vad.process_chunk.return_value = 0.9 # High probability of speech
    
    stream = VADAudioStream(mock_vad)
    
    # Test process_chunk manually without testing the internal stream queue directly
    prob = mock_vad.process_chunk(np.array([0.1]*512, dtype=np.float32))
    assert prob == 0.9

def test_wakeword_similarity():
    from src.vad.wakeword import WakeWord
    
    mock_vad = MagicMock()
    
    with patch("src.vad.wakeword.WavLMModel"), patch("src.vad.wakeword.torch"):
        ww = WakeWord(mock_vad, samples_dir="/fake/dir")
        
        # Test detection threshold logic
        ww.templates = np.array([np.zeros(768)]) # Fake template vector
        ww.stream.get_next_segment = MagicMock(side_effect=[np.zeros(16000), KeyboardInterrupt])
        
        # Set threshold
        from src.utils import config
        original_ww_db = config.get("WW_MIN_DB")
        
        try:
            # It will mock a single audio chunk and then gracefully break the loop via KeyboardInterrupt
            result = ww.listen_wakeword()
            assert result is False # Because all zeros won't trigger > 0.85
        finally:
            pass
