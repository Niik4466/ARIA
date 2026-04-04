import os
import torch
import soundfile as sf
import json
import gc
import re
import concurrent.futures
import heapq
import threading
import logging

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Warning: qwen_tts package not found. Using dummy AutoModel for structure check.")
    from transformers import AutoModel as Qwen3TTSModel

from ..utils import Config
config = Config()

verbose_mode = config.get("verbose_mode")
FLASH_ATTENTION = config.get("FLASH_ATTENTION")

if not verbose_mode:
    logging.getLogger("qwen_tts").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)

class Qwen3_TTS:
    """
    Qwen3-TTS Module for Voice Cloning and Speech Generation.
    
    Purpose: Connect with Qwen3-TTS models for high-quality voice cloning 
    and fast speech synthesis using local weights.
    """
    def __init__(self, device="cuda:0"):
        self.device = device
        self.dtype = torch.bfloat16
        self.ref_dir = os.path.join(os.path.dirname(__file__), "ref_voice")
        self.voice_clone_prompt = None
        
        # Models are not loaded initially
        self.design_model = None
        self.clone_model = None

        # Voice is loaded only if it exists. 
        # Missing voices are handled by QwenVoiceSetup.
        wav_path = os.path.join(self.ref_dir, "ref_voice.wav")
        if os.path.exists(wav_path):
            self.clone_voice()
        else:
            print("[Qwen3_TTS] No reference voice found. Use QwenVoiceSetup to generate or clone a voice.")

    def create_voice(self, ref_text, ref_instruct, language="English"):
        """
        Creates a new voice using the Qwen3TTS-VoiceDesign model and saves artifacts to 'ref_voice'.
        Unloads models to optimize memory.
        """
        # Unload Base model if it exists to free memory
        if self.clone_model is not None:
            print("Unloading Qwen3-TTS Base model to free memory...")
            self.clone_model = None
            self.voice_clone_prompt = None # Invalidate prompt as context changes
            gc.collect()
            torch.cuda.empty_cache()

        print("Loading Qwen3-TTS VoiceDesign model...")
        
        load_kwargs = {
            "device_map": self.device,
            "dtype": self.dtype,
        }
        if FLASH_ATTENTION:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            
        self.design_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            **load_kwargs
        )

        print(f"Generating voice design for setup...")
        ref_wavs, sr = self.design_model.generate_voice_design(
            text=ref_text,
            language=language,
            instruct=ref_instruct
        )

        # Save audio (sr is in header) and text
        wav_path = os.path.join(self.ref_dir, "ref_voice.wav")
        # Optional metadata
        meta_path = os.path.join(self.ref_dir, "metadata.json")

        sf.write(wav_path, ref_wavs[0], sr)
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"sr": sr, "language": language, "instruct": ref_instruct, "text": ref_text}, f)
            
        print(f"Reference voice saved to {self.ref_dir}")

        # Unload VoiceDesign model
        print("Unloading Qwen3-TTS VoiceDesign model...")
        self.design_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def clone_voice(self):
        """
        Loads the reference voice from 'ref_voice' and prepares the cloning prompt.
        Loads the Base model if not already loaded.
        """
        # Ensure VoiceDesign is unloaded (sanity check)
        if self.design_model is not None:
            self.design_model = None
            gc.collect()
            torch.cuda.empty_cache()

        # Load Base model if not present
        if self.clone_model is None:
            print("Loading Qwen3-TTS Base model for cloning...")
            
            load_kwargs = {
                "device_map": self.device,
                "dtype": self.dtype,
            }
            if FLASH_ATTENTION:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                
            self.clone_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                **load_kwargs
            )

        wav_path = os.path.join(self.ref_dir, "ref_voice.wav")
        meta_path = os.path.join(self.ref_dir, "metadata.json")
        
        if not os.path.exists(wav_path) or not os.path.exists(meta_path):
            print("[Qwen3_TTS] Reference files missing. Cannot complete clone_voice process.")
            return

        # Load audio and SR
        ref_audio_data, ref_sr = sf.read(wav_path)
        
        # Load metadata (including text)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            ref_text = metadata.get("text", "")

        print("Creating voice clone prompt...")
        # Create prompt for the base model
        self.voice_clone_prompt = self.clone_model.create_voice_clone_prompt(
            ref_audio=(ref_audio_data, ref_sr),
            ref_text=ref_text,
        )
        print("Voice clone prompt ready.")

    def _clean_text(self, text):
        """
        Cleans text for TTS: removes emojis, markdown symbols, and think tags.
        Only keeps natural language and basic punctuation.
        """
        if not text: return ""
        
        # 1. Clean think tags (if not already handled)
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think\s*/?>", "", text, flags=re.IGNORECASE)

        # 2. Remove emojis and special pictographs 
        # (This covers most emojis in the SMP range and some BMP symbols)
        text = re.sub(r'[^\x00-\x7F\u00C0-\u017F\u2010-\u201f.,!?;:()\- ]', '', text)
        
        # 3. Remove markdown specific symbols like *, _, #, `, ~
        text = re.sub(r'[*_#`~]', '', text)
        
        # 4. Collapse multiple spaces and trim
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def generate_speech(self, text, languaje="English"):
        """
        Generates speech using the prepared clone prompt.
        """
        if self.voice_clone_prompt is None or self.clone_model is None:
            self.clone_voice()
            
        text = self._clean_text(text)
        if not text:
            return None, None
            
        print(f"Preparing speech for: {text[:30]}...")
        
        wavs, sr = self.clone_model.generate_voice_clone(
            text=text,
            language=languaje,
            voice_clone_prompt=self.voice_clone_prompt,
        )

        return wavs[0], sr

    def generate_speech_stream(self, text_stream, languaje="English"):
        """
        Receives an iterable of text chunks (stream).
        Generates audio in parallel using a priority queue to maintain order.
        Yields (wav, sr) for each processed audio chunk in the correct sequence.
        """

        if self.voice_clone_prompt is None or self.clone_model is None:
            self.clone_voice()

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
        # max_workers=3 is a balanced choice for single GPU to avoid excessive memory usage
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        def worker(sentence_id, text):
            try:
                # Actual audio generation
                wavs, sr = self.clone_model.generate_voice_clone(
                    text=text,
                    language=languaje,
                    voice_clone_prompt=self.voice_clone_prompt,
                )
                result = (sentence_id, wavs[0], sr)
            except Exception as e:
                print(f"[TTS Worker] Error generating audio for ID {sentence_id}: {e}")
                # Send None results to allow the queue to progress in case of error
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
                
                accumulator += content
                parts = sentence_end_pattern.split(accumulator)
                
                # parts = [sentence1, delimiter1, sentence2, delimiter2, ..., remainder]
                while len(parts) > 1:
                    sentence_with_delim = parts.pop(0) + parts.pop(0)
                    sentence = self._clean_text(sentence_with_delim)
                    if sentence:
                        print(f"[TTS Producer] Submitting ID {next_id_to_assign}: {sentence[:30]}...")
                        executor.submit(worker, next_id_to_assign, sentence)
                        next_id_to_assign += 1
                
                accumulator = parts[0]
            
            # Process remaining text in accumulator
            final_sentence = self._clean_text(accumulator.strip())
            if final_sentence:
                print(f"[TTS Producer] Submitting final ID {next_id_to_assign}: {final_sentence[:30]}...")
                executor.submit(worker, next_id_to_assign, final_sentence)
                next_id_to_assign += 1
            
            with pq_condition:
                text_stream_finished = True
                pq_condition.notify_all()

        # Start producer thread to consume text_stream and dispatch workers
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()
        
        try:
            # Consumer loop (yields results in order)
            while True:
                with pq_condition:
                    # Wait logic:
                    # Continue if:
                    # 1. Priority Queue has the next expected ID
                    # 2. OR we have processed everything and the producer is finished
                    while not (pq and pq[0][0] == next_id_to_yield) and \
                        not (text_stream_finished and next_id_to_yield >= next_id_to_assign):
                        pq_condition.wait(timeout=0.05)
                    
                    # Exit condition
                    if text_stream_finished and next_id_to_yield >= next_id_to_assign:
                        break
                    
                    # Order check: only unqueue if the head matches our 'next_id_to_yield'
                    if pq and pq[0][0] == next_id_to_yield:
                        item_id, wav, sr = heapq.heappop(pq)
                        next_id_to_yield += 1
                    else:
                        continue # Should not happen with current wait logic, but for safety
                        
                # Yield outside the lock to allow other threads to push results
                if wav is not None:
                    yield wav, sr
        finally:
            # Cleanup
            producer_thread.join(timeout=1)
            executor.shutdown(wait=False)

class QwenVoiceSetup:
    """
    Handles initial configuration, generation, and cloning for Qwen3-TTS voices.
    """
    def __init__(self, tts_instance, asr_instance=None):
        self.tts = tts_instance
        self.asr = asr_instance
        self.ref_dir = self.tts.ref_dir
        
        # Create a default voice if it doesn't exist
        wav_path = os.path.join(self.ref_dir, "ref_voice.wav")
        if not os.path.exists(self.ref_dir) or not os.path.exists(wav_path):
            print(f"[QwenVoiceSetup] Directory '{self.ref_dir}' missing or empty. Creating default voice...")
            os.makedirs(self.ref_dir, exist_ok=True)
            self._create_default_voice()

    def _create_default_voice(self):
        self.Change_Voice(
            text="Hello! I am ready to help you with whatever you need. If you have any questions, doubts, or just want to chat, I will be here for you.",
            instruct="Female, 24 years old, confident and friendly voice - articulate pronunciation, warm tone, steady pace with a professional yet approachable demeanor.",
            language="English"
        )

    def Change_Voice(self, text, instruct, language="English"):
        """
        Generates a new reference voice based on textual instruction.
        """
        print("[QwenVoiceSetup] Generating new voice based on instructions...")
        self.tts.create_voice(
            ref_text=text,
            ref_instruct=instruct,
            language=language
        )
        self.tts.clone_voice()
        print("[QwenVoiceSetup] Voice successfully generated and changed.")

    def Re_Generate_Voice(self):
        """
        Re-generates the voice based on stored metadata instructions.
        """
        print("[QwenVoiceSetup] Regenerating voice from stored metadata...")
        meta_path = os.path.join(self.ref_dir, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"[QwenVoiceSetup] Error: Metadata not found at {meta_path}. Cannot regenerate.")
            return False
            
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[QwenVoiceSetup] Error reading metadata: {e}")
            return False
            
        ref_text = metadata.get("text", "")
        ref_instruct = metadata.get("instruct", "")
        language = metadata.get("language", "English")
        
        if not ref_instruct:
            print("[QwenVoiceSetup] Error: No generation instructions found in metadata.")
            return False
            
        try:
            self.Change_Voice(ref_text, ref_instruct, language)
            return True
        except Exception as e:
            print(f"[QwenVoiceSetup] Error changing voice: {e}")
            if "CUDA out of memory" in str(e):
                return "CUDA out of memory"
            return False

    def Clone_Voice(self, audio_data, sample_rate=16000, language="English"):
        """
        Clones a voice from provided audio data.
        Requires ASR backend to generate the base text.
        """
        if self.asr is None:
            print("[QwenVoiceSetup] Error: ASR module is required to clone voice from audio.")
            return False
            
        print("[QwenVoiceSetup] Transcribing audio for voice cloning...")
        ref_text = self.asr.speech_to_text(audio_data)
        
        if not ref_text:
            print("[QwenVoiceSetup] Error: Failed to transcribe audio. Ensure audio contains clear speech.")
            return False
            
        print(f"[QwenVoiceSetup] Transcription successful: '{ref_text}'")
        
        wav_path = os.path.join(self.ref_dir, "ref_voice.wav")
        meta_path = os.path.join(self.ref_dir, "metadata.json")
        
        # Save the audio file directly
        os.makedirs(self.ref_dir, exist_ok=True)
        sf.write(wav_path, audio_data, sample_rate)
        
        # Update metadata
        metadata = {
            "sr": sample_rate,
            "language": language,
            "instruct": "Voice cloned directly from user-provided audio.",
            "text": ref_text
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
            
        print("[QwenVoiceSetup] Audio and metadata saved. Updating TTS clone prompt...")
        self.tts.clone_voice()
        print("[QwenVoiceSetup] Voice successfully cloned!")
        return True

if __name__ == "__main__":
    import sys
    import os
    import threading
    import queue
    import time
    
    # Add project root to path to allow importing src
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    from src.audio_io import player
    from src.agent import call_ollama_stream
    
    # Proof of concept: Qwen3_TTS + Streaming LLM integration testing
    tts = Qwen3_TTS()
    
    audio_queue = queue.Queue()
    metrics = {"first_chunk_played": False, "start_time": 0, "ttfa": 0}
    
    def playback_worker():
        """Consumer thread that plays audio from the queue."""
        while True:
            item = audio_queue.get()
            if item is None:
                audio_queue.task_done()
                break
            
            # Performance Metric check: Time to First Audio (TTFA)
            if not metrics["first_chunk_played"]:
                metrics["ttfa"] = time.time() - metrics["start_time"]
                metrics["first_chunk_played"] = True
                print(f"\n[METRIC] TIME TO FIRST AUDIO (TTFA): {metrics['ttfa']:.4f} seconds")
            
            wav, sr = item
            print(f"\n[Playback] Playing audio chunk ({len(wav)/sr:.2f} s)...")
            player.play(wav, sr)
            audio_queue.task_done()

    # Start playback thread
    pb_thread = threading.Thread(target=playback_worker, daemon=True)
    pb_thread.start()
    
    prompt = "Explica de forma breve que es un agujero negro."
    print(f"\n--- Testing INTEGRATION LLM (stream) + TTS (sentences) for: '{prompt}' ---")
    
    try:
        metrics["start_time"] = time.time()
        
        # 1. Get text generator from the agent
        text_gen = call_ollama_stream(prompt)
        
        # 2. Pass generator to TTS engine
        # We put results in the queue for the worker to play in parallel
        for wav, sr in tts.generate_speech_stream(text_gen, languaje="Spanish"):
            print(f"[Engine] Audio generated, sending to queue...")
            audio_queue.put((wav, sr))
        
        # Close queue and wait for completion
        print("\n--- End of text and audio generation, waiting for playback to finish ---")
        audio_queue.put(None)
        audio_queue.join()
        pb_thread.join()
        print("--- Test completed successfully ---")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()