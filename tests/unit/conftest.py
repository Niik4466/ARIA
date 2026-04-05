import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

@pytest.fixture(autouse=True)
def mock_heavy_dependencies():
    """
    Automatically mock heavy AI and IO dependencies for every unit test
    to guarantee test speed and prevent CUDA out-of-memory issues natively.
    """
    # Mocks that apply universally: Deep model dependencies rather than module wrappers
    mocks = [
        patch("faster_whisper.WhisperModel"),
        patch("chromadb.PersistentClient"),
        patch("sounddevice.play"),
        patch("sounddevice.wait"),
        patch("sounddevice.stop"),
        patch("src.vad.wakeword.WavLMModel"),
        patch("src.agent.call_ollama"),
        patch("src.agent.call_ollama_stream")
    ]
    
    # Start all mock patches
    started_mocks = [m.start() for m in mocks]
    
    yield
    
    # Stop all patches after the test
    for m in mocks:
        m.stop()

@pytest.fixture
def clean_config():
    """
    Fixture returning a fresh instance of config without caching.
    Useful for test_utils.py.
    """
    from src.utils import Config
    old_instance = Config._instance
    Config._instance = None # Force recreation
    
    # Mock file-system specific config checks internally if needed later
    yield
    
    Config._instance = old_instance
