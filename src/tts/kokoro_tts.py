import os
import torch
import numpy as np
import soundfile as sf
import gc
import re
import concurrent.futures
import heapq
import threading
from src.agent import clean_emojis
from typing import Optional

try:
    from kokoro import KPipeline
except ImportError:
    print("Warning: kokoro package not found. Please install it with 'pip install kokoro'.")
    class KPipeline:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return []

from ..utils import Config
config = Config()

KOKORO_LANG = config.get("KOKORO_LANG")
KOKORO_VOICE = config.get("KOKORO_VOICE")
KOKORO_SPEED = config.get("KOKORO_SPEED")

verbose = config.get("verbose_mode")
_builtins_print = print
def print(*args, **kwargs):
    if verbose:
        _builtins_print(*args, **kwargs)

class Kokoro_TTS:
    """
    Backend for Kokoro TTS, similar to the structure of Qwen3_TTS.
    Allows high-quality audio generation using the Kokoro model.
    """
    def __init__(self, device: Optional[str] = None):
        """
        Initializes the Kokoro TTS engine.
        
        Args:
            device (str, optional): Device for the model ('cuda' or 'cpu'). If None, it auto-detects.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.lang = KOKORO_LANG
        self.voice_name = KOKORO_VOICE
        self.speed = KOKORO_SPEED
        
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """
        Loads the Kokoro KPipeline model into memory.
        """
        print(f"Loading Kokoro TTS model (lang={self.lang}) on {self.device}...")
        try:
            self.pipeline = KPipeline(lang_code=self.lang, device=self.device)
            print("Kokoro TTS model loaded successfully.")
        except Exception as e:
            print(f"Error loading Kokoro TTS: {e}")
            self.pipeline = None

    def unload_model(self):
        """
        Releases model memory.
        """
        print("Unloading Kokoro TTS model from memory...")
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Memory released.")

    def _clean_text(self, text):
        """
        Cleans text for TTS: removes emojis, markdown symbols, and think tags.
        """
        if not text: return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think\s*/?>", "", text, flags=re.IGNORECASE)
        text = re.sub(r'[^\x00-\x7F\u00C0-\u017F\u2010-\u201f.,!?;:()\- ]', '', text)
        text = re.sub(r'[*_#`~]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def generate_speech(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None, languaje: str = None):
        """
        Generates speech from text.
        
        Args:
            text (str): Text to convert to audio.
            voice (str, optional): Voice name (e.g., 'af_sarah'). If None, uses value from config.py.
            speed (float, optional): Speed factor. If None, uses value from config.py.
            
        Returns:
            tuple: (audio_numpy_array, sample_rate)
        """
        if self.pipeline is None:
            self.load_model()
            if self.pipeline is None:
                print("Error: Could not initialize Kokoro pipeline.")
                return None, 24000
                
        target_voice = voice if voice is not None else self.voice_name
        target_speed = speed if speed is not None else self.speed
        
        print(f"Generating audio with Kokoro for: {text[:50]}...")
        
        try:
            # KPipeline returns a generator of (graphemes, phonemes, audio)
            generator = self.pipeline(
                text, 
                voice=target_voice, 
                speed=target_speed, 
                split_pattern=r'\n+'
            )
            
            chunks = []
            for gs, ps, audio in generator:
                if audio is not None:
                    chunks.append(audio)
            
            if not chunks:
                print("Kokoro did not generate any audio chunks.")
                return None, 24000
                
            audio_full = np.concatenate(chunks)
            return audio_full, 24000
            
        except Exception as e:
            print(f"Error during Kokoro generation: {e}")
            return None, 24000

    def generate_speech_stream(self, text_stream, voice: Optional[str] = None, speed: Optional[float] = None, languaje: str = None):
        """
        Receives an iterable of text chunks (stream).
        Generates audio in parallel using a priority queue to maintain order.
        Yields (wav, sr) for each processed audio chunk in the correct sequence.
        """
        if self.pipeline is None:
            self.load_model()

        target_voice = voice if voice is not None else self.voice_name
        target_speed = speed if speed is not None else self.speed

        # Regex to detect end of sentences (., !, ?, \n)
        sentence_end_pattern = re.compile(r'([.!?]+(?:\s+|\n|$))')
        
        # Priority Queue to store (id, wav, sr)
        pq = []
        pq_lock = threading.Lock()
        pq_condition = threading.Condition(pq_lock)
        
        # Counters
        next_id_to_assign = 0
        next_id_to_yield = 0
        
        # Control flags
        text_stream_finished = False
        
        # Thread pool for parallel generation
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        def worker(sentence_id, text):
            try:
                # Actual audio generation using provide generate_speech logic but for a single chunk
                generator = self.pipeline(
                    text, 
                    voice=target_voice, 
                    speed=target_speed, 
                    split_pattern=r'\n+'
                )
                
                chunks = []
                for gs, ps, audio in generator:
                    if audio is not None:
                        chunks.append(audio)
                
                if chunks:
                    wav = np.concatenate(chunks)
                    result = (sentence_id, wav, 24000)
                else:
                    result = (sentence_id, None, None)
            except Exception as e:
                print(f"[Kokoro Worker] Error generating audio for ID {sentence_id}: {e}")
                result = (sentence_id, None, None)
            
            with pq_condition:
                heapq.heappush(pq, result)
                pq_condition.notify_all()

        def producer():
            nonlocal next_id_to_assign, text_stream_finished
            accumulator = ""
            
            for chunk in text_stream:
                if isinstance(chunk, dict):
                    content = chunk.get("response", "")
                else:
                    content = str(chunk)
                
                content = clean_emojis(content)
                accumulator += content
                parts = sentence_end_pattern.split(accumulator)
                
                while len(parts) > 1:
                    sentence_with_delim = parts.pop(0) + parts.pop(0)
                    sentence = self._clean_text(sentence_with_delim)
                    if sentence:
                        print(f"[Kokoro Producer] Submitting ID {next_id_to_assign}: {sentence[:30]}...")
                        executor.submit(worker, next_id_to_assign, sentence)
                        next_id_to_assign += 1
                
                accumulator = parts[0]
            
            final_sentence = self._clean_text(accumulator.strip())
            if final_sentence:
                print(f"[Kokoro Producer] Submitting final ID {next_id_to_assign}: {final_sentence[:30]}...")
                executor.submit(worker, next_id_to_assign, final_sentence)
                next_id_to_assign += 1
            
            with pq_condition:
                text_stream_finished = True
                pq_condition.notify_all()

        # Start producer thread
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()
        
        try:
            while True:
                with pq_condition:
                    while not (pq and pq[0][0] == next_id_to_yield) and \
                        not (text_stream_finished and next_id_to_yield >= next_id_to_assign):
                        pq_condition.wait(timeout=0.05)
                    
                    if text_stream_finished and next_id_to_yield >= next_id_to_assign:
                        break
                    
                    if pq and pq[0][0] == next_id_to_yield:
                        item_id, wav, sr = heapq.heappop(pq)
                        next_id_to_yield += 1
                    else:
                        continue 
                        
                if wav is not None:
                    yield wav, sr
        finally:
            producer_thread.join(timeout=1)
            executor.shutdown(wait=False)

    def unload_model(self):
        """
        Releases model memory.
        """
        print("Unloading Kokoro TTS model from memory...")
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Memory released.")

    def save_audio(self, audio: np.ndarray, sr: int, filename: str = "output_kokoro.wav"):
        """
        Saves the generated audio to a file.
        """
        if audio is not None:
            sf.write(filename, audio, sr)
            print(f"Audio saved to: {filename}")
        else:
            print("No audio available to save.")
