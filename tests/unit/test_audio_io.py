import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from unittest.mock import patch, MagicMock

@patch("src.audio_io.sd")
def test_audio_player_sync_playback(mock_sd):
    from src.audio_io import AudioPlayer
    player = AudioPlayer()
    
    # Fake audio
    audio = np.array([0.1, 0.2, -0.1], dtype=np.float32)
    sr = 16000
    
    player.play(audio, sr, blocking=True)
    
    # Assert play and block
    mock_sd.play.assert_called_once()
    mock_sd.wait.assert_called_once()
    assert np.array_equal(mock_sd.play.call_args[0][0], audio)

@patch("src.audio_io.sd")
def test_audio_player_resampling(mock_sd):
    from src.audio_io import AudioPlayer
    player = AudioPlayer()
    
    audio = np.array([0.1, 0.2, -0.1], dtype=np.float32)
    sr = 22050 # Not in default supported rates
    
    # Setup mock to fail first check
    mock_sd.check_output_settings.side_effect = Exception("Not supported")
    
    player.play(audio, sr, blocking=False)
    
    # Assert resampling triggered (output should be 48000)
    mock_sd.play.assert_called_once()
    # Check that sample rate was transformed to 48000
    assert mock_sd.play.call_args[0][1] == 48000

@patch("src.audio_io.sd.stop")
def test_audio_player_stop(mock_sd_stop):
    from src.audio_io import AudioPlayer
    player = AudioPlayer()
    
    # Fill queue
    player.play_async(np.array([0.1]), 16000)
    
    player.stop()
    
    # Assure sounddevice was killed
    mock_sd_stop.assert_called_once()
    # Assert queue was cleared
    assert player._audio_queue.empty() is True
