import torch
import numpy as np
import platform
import subprocess
import threading
import queue
import sys
import time
from collections import deque
from src.utils import Config
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
    Enhanced audio stream with pre-roll buffer and flexible silence detection 
    utilizing FFmpeg as an intermediary layer. This solves hardware limitations
    by pushing resampling tasks to FFmpeg directly.
    """
    def __init__(self, vad_instance):
        self.vad_iterator = vad_instance.get_iterator()
        self.window_size_samples = 512
        self.q = queue.Queue()
        self.process = None
        self.capture_thread = None
        self.running = False
        
        # Pre-roll buffer (approx 0.5s of constant audio to avoid losing initial syllables)
        self.pre_roll_len = int((SAMPLE_RATE * 0.5) / self.window_size_samples)
        self.pre_roll_buffer = deque(maxlen=self.pre_roll_len)

    def _capture_loop(self):
        """Continuously reads float32 audio chunks from FFmpeg stdout."""
        bytes_per_sample = 4 # float32
        chunk_bytes_size = self.window_size_samples * bytes_per_sample
        
        def read_exactly(stream, size):
            buf = b''
            while len(buf) < size:
                chunk_data = stream.read(size - len(buf))
                if not chunk_data:
                    break
                buf += chunk_data
            return buf
        
        try:
            while self.running and self.process and self.process.poll() is None:
                raw_data = read_exactly(self.process.stdout, chunk_bytes_size)
                if not raw_data or len(raw_data) < chunk_bytes_size:
                    break
                
                # Convert bytes to numpy float32 array
                chunk = np.frombuffer(raw_data, dtype=np.float32).copy()
                self.q.put(chunk)
                
                # Always maintain some history for the pre-roll
                self.pre_roll_buffer.append(chunk)
        except Exception as e:
            print(f"Error in audio capture loop: {e}", file=sys.stderr)
        finally:
            self.stop()

    def start(self):
        """Starts the audio stream using FFmpeg subprocess for broader hardware compatibility."""
        if self.running:
            return
            
        system = platform.system()
        # Default options for FFmpeg depending on the OS
        if system == "Linux":
            input_format = "pulse"
            input_device = "default"
        elif system == "Darwin":
            input_format = "avfoundation"
            input_device = ":0"
        elif system == "Windows":
            input_format = "dshow"
            input_device = "audio=Microphone" # NOTE: Requires specific setup or modification per machine
        else:
            input_format = "pulse"
            input_device = "default"

        ffmpeg_cmd = [
            "ffmpeg",
            "-f", input_format,
            "-i", input_device,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "f32le",
            "-"
        ]
        
        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, # Ignore FFmpeg logs
                bufsize=10**8
            )
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
        except FileNotFoundError:
            print("[Error] FFmpeg is not installed or not found in system PATH. Please install FFmpeg.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Failed to start FFmpeg stream: {e}", file=sys.stderr)
            sys.exit(1)

    def stop(self):
        """Stops the FFmpeg process and closes the stream."""
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        if self.capture_thread and self.capture_thread.is_alive():
            if threading.current_thread() != self.capture_thread:
                self.capture_thread.join(timeout=1.0)

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
                        flat_audio = np.concatenate(audio_buffer).flatten()
                        # Clear buffer for next speech
                        audio_buffer = []
                        # Ignore clicks/noise shorter than 0.8 seconds (12800 samples)
                        if len(flat_audio) < int(SAMPLE_RATE * 0.8):
                            continue
                        return flat_audio
            
            if in_speech:
                audio_buffer.append(chunk)
                # 20s safety limit
                if len(audio_buffer) * self.window_size_samples > 20 * SAMPLE_RATE:
                    print(" [ASR] (Voice segment too long, cutting) ")
                    return np.concatenate(audio_buffer).flatten()

        return None
