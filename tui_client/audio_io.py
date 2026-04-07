import sounddevice as sd
import numpy as np
import threading
import queue

from src.utils import Config
config = Config()

SUPPORTED_SAMPLE_RATES = [48000, 44100, 16000, 8000]

def _resample_audio(audio, orig_sr, target_sr):
    """Resample audio using scipy or fallback to linear interpolation."""
    try:
        from scipy.signal import resample
        num_samples = int(len(audio) * target_sr / orig_sr)
        return resample(audio, num_samples)
    except ImportError:
        # Fallback: simple linear interpolation
        num_samples = int(len(audio) * target_sr / orig_sr)
        indices = np.linspace(0, len(audio) - 1, num_samples)
        return np.interp(indices, np.arange(len(audio)), audio)

verbose_mode = config.get("verbose_mode")
_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)

class AudioPlayer:
    """
    Backend for playing audio produced by TTS modules.
    Provides a simple interface to play NumPy arrays synchronously or asynchronously.
    Uses a queue and a worker thread to manage concurrent or asynchronous playbacks smoothly.
    """
    def __init__(self):
        self._audio_queue = queue.Queue()
        self._playback_thread = threading.Thread(target=self._worker, daemon=True)
        self._playback_thread.start()

    def _worker(self):
        """Background thread that consumes from the queue and plays audio sequentially."""
        while True:
            item = self._audio_queue.get()
            if item is None:
                break
                
            audio, sr = item
            # Play in blocking mode within this separate thread
            self.play(audio, sr, blocking=True)
            self._audio_queue.task_done()

    def play(self, audio, sr, blocking=True):
        """
        Plays a NumPy audio array.
        
        Args:
            audio (np.ndarray): The audio array (float32 or int16).
            sr (int): Sample Rate.
            blocking (bool): If True, the function waits for playback to finish.
        """
        if audio is None or len(audio) == 0:
            print("[AudioPlayer] Error: Audio is empty or None.")
            return

        # Basic normalization if audio is not float32
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767.0
            else:
                # Generic conversion attempt
                audio = audio.astype(np.float32)

        try:
            sr = int(sr)
            # Check if sample rate is supported by the device
            if sr not in SUPPORTED_SAMPLE_RATES:
                try:
                    sd.check_output_settings(samplerate=sr)
                except Exception:
                    print(f"[AudioPlayer] Sample rate {sr} not supported, resampling to 48000...")
                    audio = _resample_audio(audio, sr, 48000)
                    sr = 48000
            sd.play(audio, sr)
            if blocking:
                sd.wait()
        except Exception as e:
            print(f"[AudioPlayer] ❌ Error during playback: {e}")

    def play_async(self, audio, sr):
        """
        Queues audio to be played in the background thread sequentially.
        """
        if audio is None or len(audio) == 0:
            print("[AudioPlayer] Warning: Attempting to play empty audio async.")
            return
            
        self._audio_queue.put((audio, sr))

    def stop(self):
        """
        Stops any active playback and clears all queued pending audios.
        """
        # Empty the queue so we don't play next items automatically
        with self._audio_queue.mutex:
            self._audio_queue.queue.clear()
            
        # Stop sounddevice right away (will instantly unblock sd.wait() in the worker thread)
        sd.stop()

# Default global instance
player = AudioPlayer()
