import os
import torch
import numpy as np
import sys
import soundfile as sf
import warnings
from transformers import AutoFeatureExtractor, WavLMModel
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import Config
config = Config()

SAMPLE_RATE = config.get("SAMPLE_RATE")
WAKEWORD_SAMPLES = config.get("WAKEWORD_SAMPLES")
WAKEWORD_THRESHOLD = config.get("WAKEWORD_THRESHOLD")
verbose_mode = config.get("verbose_mode")

_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)

# Suppress certain warnings
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")

# Import VAD classes from the sibling file
from .vad import VAD, VADAudioStream

class WakeWordSetup:
    """
    Handles initial configuration and sample registration for the wakeword.
    """
    def __init__(self, vad_instance, samples_dir=None, tts=None, rvc=None, audio_player=None):
        self.vad = vad_instance
        self.tts = tts
        self.rvc = rvc
        self.audio_player = audio_player
        
        if samples_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.samples_dir = os.path.join(current_dir, "samples")
        else:
            self.samples_dir = samples_dir
            if not os.path.isabs(self.samples_dir):
                self.samples_dir = os.path.join(os.getcwd(), self.samples_dir)

        self.stream = VADAudioStream(self.vad)
        self.setup_history = ""

    def _speak(self, instruction):
        QWEN3_LANG = config.get("QWEN3_LANG")
        from src.agent import call_ollama, clean_think_tags, clean_emojis, get_wakeword_prompt
        
        lang = QWEN3_LANG
        sys_prompt = get_wakeword_prompt(language=lang, history=self.setup_history)
        
        try:
            # Setting temperature to 0.7 to encourage non-repetitive phrases over time
            text = call_ollama(prompt=instruction, system_prompt=sys_prompt, temperature=0.7)
            text = clean_think_tags(text)
            text = clean_emojis(text)
            print(f"[🤖 ARIA] {text}")
            
            self.setup_history += f"- Assistant said: {text}\n"
            
            if self.tts and self.audio_player:
                wav, sr = self.tts.generate_speech(text, languaje=lang)
                if wav is not None:
                    if self.rvc:
                        wav, sr = self.rvc.transform_numpy(wav, sr)
                    self.audio_player.play(wav, sr)
        except Exception as e:
            print(f"[System] ⚠️ Error in TTS during setup: {e}")

    def has_enough_samples(self):
        """Checks if the samples directory contains enough .wav files."""
        if not os.path.exists(self.samples_dir):
            return False
        wav_files = [f for f in os.listdir(self.samples_dir) if f.endswith(".wav")]
        return len(wav_files) >= WAKEWORD_SAMPLES

    def new_wakeword_samples(self):
        """
        Deletes existing samples and records new ones for a new wakeword.
        """
        print("\n=== NEW WAKEWORD SETUP ===")
        is_first_time = not os.path.exists(self.samples_dir)
        
        if not is_first_time:
            import shutil
            shutil.rmtree(self.samples_dir)
            
        os.makedirs(self.samples_dir, exist_ok=True)
        
        if is_first_time:
            self._speak(f"Introduce yourself to the user and welcome them. Then briefly explain that we are going to set up their activation word. Explain that you will ask the user for it {WAKEWORD_SAMPLES} times.")
        else:
            self._speak(f"Greets and briefly explain that we are going to set up a new activation word (wakeword) to call you, make the user understand that you will ask for the activation word {WAKEWORD_SAMPLES} times")
        
        self.add_wakeword_samples(WAKEWORD_SAMPLES)
        if is_first_time:
            self._speak("Explain that everything is ready and the activation word was successfully learned.")
        print("=== Configuration completed successfully! ===\n")

    def add_wakeword_samples(self, num_samples=1):
        """
        Records and adds one or more wakeword samples to the directory.
        """
        if not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)
            
        current_files = [f for f in os.listdir(self.samples_dir) if f.endswith(".wav")]
        current_count = len(current_files)
            
        self.stream.start()
        try:
            count = 0
            while count < num_samples:
                sample_num = current_count + count + 1
                
                print(f"\nSample {count + 1}/{num_samples} (Total: {sample_num}/{WAKEWORD_SAMPLES})")
                if count == num_samples - 1:
                    self._speak("Asks the user to say your activation word one last time to finish")
                elif count == 0:
                    self._speak("Asks the user to say your activation word")
                else:
                    self._speak("Asks the user to say your activation word again")
                
                self.stream.clear_queue()
                print("Listening...")
                audio = self.stream.get_next_segment(timeout=15.0)
                
                if audio is None or len(audio) == 0:
                    print("❌ Empty audio or error. Try again.")
                    self._speak("Apologize and say you didn't hear them well, and ask them to try again.")
                    continue
                
                # Save to wav
                file_path = os.path.join(self.samples_dir, f"{sample_num}.wav")
                sf.write(file_path, audio, SAMPLE_RATE)
                print(f"✅ Sample {sample_num} saved.")
                count += 1
                
        finally:
            self.stream.stop()
            self.setup_history = ""

class WakeWord:
    """
    WakeWord Class for Wakeword Detection.
    Handles continuous listening using saved templates.
    """
    def __init__(self, vad_instance, samples_dir=None, device="cpu"):
        self.device = device
        self.vad = vad_instance
        
        # Set samples_dir relative to this script if not provided
        if samples_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.samples_dir = os.path.join(current_dir, "samples")
        else:
            self.samples_dir = samples_dir
            if not os.path.isabs(self.samples_dir):
                self.samples_dir = os.path.join(os.getcwd(), self.samples_dir)

        self.templates = []
        print(f"[WakeWord] Samples directory: {self.samples_dir}")

        print("[WakeWord] Loading WavLM model...")
        self.processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.model.to(self.device)
        self.model.eval()

        # Initialize audio stream with the shared VAD instance
        self.stream = VADAudioStream(self.vad)
        
        self._load_templates()

    def _extract_embedding(self, audio):
        """Extracts a normalized embedding from an audio segment using WavLM."""
        if len(audio) < 1000:
            return None
        
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Simple Mean Pooling
        embedding = torch.mean(hidden_states, dim=1)
        embedding = embedding.cpu().numpy()
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

    def _load_templates(self):
        """Loads embeddings (templates) from the saved .wav files."""
        print(f"[WakeWord] Loading templates from {self.samples_dir}...")
        self.templates = []
        for i in range(1, WAKEWORD_SAMPLES + 1):
            file_path = os.path.join(self.samples_dir, f"{i}.wav")
            if os.path.exists(file_path):
                audio, sr = sf.read(file_path)
                emb = self._extract_embedding(audio)
                if emb is not None:
                    self.templates.append(emb)
        
        if self.templates:
            self.templates = np.vstack(self.templates)
            print(f"[WakeWord] {len(self.templates)} templates loaded.")
        else:
            print("[WakeWord] ❌ No templates loaded.")

    def listen_wakeword(self):
        """
        Continuous listening loop. 
        Returns True when the wakeword is detected.
        """
        if not self.templates.any():
            print("[WakeWord] Error: No templates loaded for detection.")
            return False

        print("\n[WakeWord] Starting continuous listening... (Ctrl+C to stop)")
        self.stream.start()
        self.stream.clear_queue()

        try:
            while True:
                audio = self.stream.get_next_segment()
                if audio is None:
                    continue
                
                emb = self._extract_embedding(audio)
                if emb is not None:
                    # Calculate similarity with all templates
                    scores = cosine_similarity(emb, self.templates)[0]
                    max_score = np.max(scores)
                    
                    # Detection requires at least 2 positive votes (similarity > threshold)
                    votes = np.sum(scores > WAKEWORD_THRESHOLD)
                    
                    if votes >= 2:
                        print(f"\n✨ WAKEWORD DETECTED! (Score: {max_score:.3f}, Votes: {votes}/{len(self.templates)}) ✨\n")
                        self.stream.stop()
                        return True
                    else:
                        # Optional: debug output
                        # print(f"Score: {max_score:.3f} | Votes: {votes}")
                        pass
        except KeyboardInterrupt:
            print("\n[WakeWord] Stopping...")
            self.stream.stop()
        
        return False

if __name__ == "__main__":
    # Test execution
    from .vad import VAD # Local import for testing
    vad = VAD()
    ww = WakeWord(vad)
    if ww.listen_wakeword():
        print("Success: Wakeword triggered!")
    else:
        print("WakeWord stopped.")
