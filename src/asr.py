import time
import numpy as np
from faster_whisper import WhisperModel
from .utils import Config
from src.logger import mlog as print

config = Config()

ASR_MODEL_ID = config.get("ASR_MODEL_ID")
ASR_DEVICE = config.get("ASR_DEVICE")
ASR_COMPUTE_TYPE = config.get("ASR_COMPUTE_TYPE")
ASR_LANGUAGE = config.get("ASR_LANGUAGE")
ASR_BEAM_SIZE = config.get("ASR_BEAM_SIZE")
verbose_mode = config.get("verbose_mode")

import logging
if not verbose_mode:
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

class ASR:
    """
    ASR Class for Automatic Speech Recognition.
    Uses faster-whisper for transcription and Silero VAD for voice activity detection.
    """
    def __init__(self):
        """
        Initializes the ASR module.
        """
        print(f"[ASR] Loading model {ASR_MODEL_ID} on {ASR_DEVICE}...")
        try:
            self.model = WhisperModel(
                ASR_MODEL_ID, 
                device=ASR_DEVICE, 
                compute_type=ASR_COMPUTE_TYPE
            )
            print("[ASR] Model loaded successfully.")
        except Exception as e:
            print(f"[ASR] ❌ Error loading model: {e}")
            raise e

    def speech_to_text(self, audio_data):
        """
        Transcribes the provided audio data to text.
        :param audio_data: Numpy array of audio samples (float32).
        :return: Transcribed text string.
        """
        if audio_data is None or len(audio_data) == 0:
            return ""

        start_t = time.time()
        try:
            # transcribe returns a generator of segments and info
            segments, info = self.model.transcribe(
                audio_data, 
                language=ASR_LANGUAGE,
                beam_size=ASR_BEAM_SIZE,
                vad_filter=False # We already filtered with VAD
            )
            
            # Combine all detected segments into a single string
            text = " ".join([seg.text for seg in segments]).strip()
            
            dt = time.time() - start_t
            if text:
                print(f"[ASR] Transcribed ({dt:.2f}s): {text}")
            
            return text
            
        except Exception as e:
            print(f"[ASR] ❌ Error in transcription: {e}")
            return ""

if __name__ == "__main__":
    pass
