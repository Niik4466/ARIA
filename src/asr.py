import time
import numpy as np
from faster_whisper import WhisperModel
from .vad.vad import VADAudioStream
from .utils import Config
config = Config()

ASR_MODEL_ID = config.get("ASR_MODEL_ID")
ASR_DEVICE = config.get("ASR_DEVICE")
ASR_COMPUTE_TYPE = config.get("ASR_COMPUTE_TYPE")
ASR_LANGUAGE = config.get("ASR_LANGUAGE")
ASR_BEAM_SIZE = config.get("ASR_BEAM_SIZE")
verbose_mode = config.get("verbose_mode")

_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)

import logging
if not verbose_mode:
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)


class ASR:
    """
    ASR Class for Automatic Speech Recognition.
    Uses faster-whisper for transcription and Silero VAD for voice activity detection.
    """
    def __init__(self, vad_instance):
        """
        Initializes the ASR module.
        :param vad_instance: An instance of the VAD class (from vad.py).
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

        # Initialize the audio stream with the shared VAD instance
        self.stream = VADAudioStream(vad_instance)

    def listen(self, timeout=10.0):
        """
        Listens for user speech using the VAD-enabled audio stream.
        :param timeout: Maximum time to wait for speech to start.
        :return: Numpy array of audio data or None if no speech detected.
        """
        print("\n🎤 [Listening...] ", end="", flush=True)
        self.stream.start()
        self.stream.clear_queue()
        
        try:
            # get_next_segment handles the VAD logic
            audio_segment = self.stream.get_next_segment(timeout=timeout)
            if audio_segment is not None:
                print(" [Speech detected] ")
            else:
                print(" [No speech detected] ")
            return audio_segment
        finally:
            self.stream.stop()

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
    # Test execution
    from .vad.vad import VAD
    import sys
    
    try:
        vad = VAD()
        asr = ASR(vad)
        
        print("\n--- ASR Test ---")
        print("Speak now...")
        
        audio = asr.listen()
        if audio is not None:
            text = asr.speech_to_text(audio)
            print(f"Result: {text}")
        else:
            print("No audio captured.")
            
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)
