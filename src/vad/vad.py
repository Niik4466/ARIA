import torch
import numpy as np
import sounddevice as sd
import queue
import sys
import time
from collections import deque
from ..utils import Config
config = Config()

SAMPLE_RATE = config.get("SAMPLE_RATE")
VAD_MIN_SILENCE_MS = config.get("VAD_MIN_SILENCE_MS")
VAD_SPEECH_PAD_MS = config.get("VAD_SPEECH_PAD_MS")
verbose_mode = config.get("verbose_mode")

_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)

class VAD:
    """
    Voice Activity Detection class powered by Silero VAD.
    """
    def __init__(self):
        print("[VAD] Loading Silero VAD...")
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', 
                model='silero_vad', 
                force_reload=False
            )
            (get_speech_timestamps, save_audio, read_audio, self.VADIterator, collect_chunks) = utils
        except Exception as e:
            print(f"Error loading Silero VAD: {e}")
            sys.exit(1)

    def get_iterator(self):
        """Returns a new VADIterator with flexible configuration."""
        return self.VADIterator(
            self.model, 
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS
        )

class VADAudioStream:
    """
    Enhanced audio stream with pre-roll buffer and flexible silence detection.
    """
    def __init__(self, vad_instance):
        self.vad_iterator = vad_instance.get_iterator()
        self.window_size_samples = 512
        self.q = queue.Queue()
        self.stream = None
        self.running = False
        
        # Pre-roll buffer (approx 0.5s of constant audio to avoid losing initial syllables)
        self.pre_roll_len = int((SAMPLE_RATE * 0.5) / self.window_size_samples)
        self.pre_roll_buffer = deque(maxlen=self.pre_roll_len)

    def _callback(self, indata, frames, time, status):
        """Audio stream callback."""
        if status:
            print(f"Audio stream status: {status}", file=sys.stderr)
        chunk = indata.copy()
        self.q.put(chunk)
        # Always maintain some history for the pre-roll
        self.pre_roll_buffer.append(chunk)

    def start(self):
        """Starts the audio stream."""
        if self.running:
            return
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            blocksize=self.window_size_samples, 
            dtype="float32", 
            callback=self._callback
        )
        self.stream.start()
        self.running = True

    def stop(self):
        """Stops and closes the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.running = False

    def clear_queue(self):
        """Clears the audio queue and resets VAD state."""
        with self.q.mutex:
            self.q.queue.clear()
        self.vad_iterator.reset_states()
        self.pre_roll_buffer.clear()

    def get_next_segment(self, timeout=10.0):
        """
        Blocks until a speech segment is detected. 
        Highly flexible for pauses and avoids sudden cut-offs.
        """
        audio_buffer = []
        in_speech = False
        start_time = time.time()
        
        while self.running:
            if time.time() - start_time > timeout and not in_speech:
                return None

            try:
                chunk = self.q.get(timeout=1.0)
            except queue.Empty:
                continue

            chunk_tensor = torch.from_numpy(chunk).squeeze()
            speech_dict = self.vad_iterator(chunk_tensor, return_seconds=False)
            
            if speech_dict:
                if 'start' in speech_dict:
                    # When detecting the start, include the pre-roll buffer
                    if not in_speech:
                        audio_buffer.extend(list(self.pre_roll_buffer))
                    in_speech = True
                
                if 'end' in speech_dict:
                    in_speech = False
                    if audio_buffer:
                        audio_buffer.append(chunk)
                        return np.concatenate(audio_buffer).flatten()
            
            if in_speech:
                audio_buffer.append(chunk)
                # 20s safety limit
                if len(audio_buffer) * self.window_size_samples > 20 * SAMPLE_RATE:
                    print(" [ASR] (Voice segment too long, cutting) ")
                    return np.concatenate(audio_buffer).flatten()

        return None
