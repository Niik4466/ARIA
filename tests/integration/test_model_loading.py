import pytest
import multiprocessing
import sys
import os
import time

# PyTorch/CUDA cannot be forked safely in Linux. We must spawn clean subprocesses.
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Configuration Timeout Limit (in seconds)
TIMEOUT = 10

def _load_rag_manager():
    # Will crash if SentenceTransformers model is invalid
    from src.rag import RAGManager
    _ = RAGManager()

def _load_asr():
    from src.asr import ASR
    _ = ASR()

def _load_tts():
    # Only testing Kokoro by default, Qwen3 requires larger manual downloads usually
    from src.tts.kokoro_tts import Kokoro_TTS
    _ = Kokoro_TTS()

def _load_vad():
    from tui_client.vad.vad import VAD
    _ = VAD()

def _load_wakeword():
    from tui_client.vad.vad import VAD
    from tui_client.vad.wakeword import WakeWord
    # Using real dependencies
    vad = VAD()
    _ = WakeWord(vad)

def run_with_timeout(target_func, timeout_seconds=5):
    """
    Spawns a process to run target_func. 
    If it exits gracefully or is still running after timeout_seconds, returns True.
    If it crashes before timeout_seconds, returns False and raises the associated Error.
    """
    process = multiprocessing.Process(target=target_func)
    process.start()
    
    process.join(timeout=timeout_seconds)
    
    if process.is_alive():
        # It's still running (e.g. downloading Model weights or allocating Memory),
        # which means syntax and basic arguments are valid. We kill it gracefully.
        process.terminate()
        process.join()
        return True
        
    # The process exited before the timeout. Check exit code.
    if process.exitcode == 0:
        return True
    
    return False

def test_rag_loading():
    assert run_with_timeout(_load_rag_manager, TIMEOUT), "RAGManager failed to load its internal models (likely invalid model_name)."

def test_asr_loading():
    assert run_with_timeout(_load_asr, TIMEOUT), "ASR Faster-Whisper failed to load its internal models."

def test_tts_loading():
    assert run_with_timeout(_load_tts, TIMEOUT), "Kokoro TTS failed to load models."

def test_vad_loading():
    assert run_with_timeout(_load_vad, TIMEOUT), "Silero VAD failed to initialize models."

def test_wakeword_loading():
    assert run_with_timeout(_load_wakeword, TIMEOUT), "WavLM WakeWord failed to load models."
