import os
import sys
import time

_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

# Override backend prints to log file
from src.logger import redirect_print_to_logger
redirect_print_to_logger()

# We keep a native print for the TUI interface
def tui_print(text, end="\n", flush=False):
    sys.stdout.write(text + end)
    if flush:
        sys.stdout.flush()

from src.container import Container
from src.graph.edges import build_graph
from src.agent import call_ollama, clean_think_tags, clean_emojis, get_acknowledgement_prompt, get_farewell_prompt
from tui_client.vad.vad import VAD, VADAudioStream
from tui_client.vad.wakeword import WakeWord, WakeWordSetup
from tui_client.audio_io import player

def tui_play_stream(audio_stream, player):
    """Consumes the generator from GraphState and writes to speaker."""
    for wav, sr in audio_stream:
        player.play(wav, sr)

def run_client():
    tui_print("\n--- ARIA TUI CLIENT READY ---")
    tui_print("Loading AI Backend...")
    container = Container()
    app = build_graph()
    
    tui_print("Loading Audio Drivers...")
    vad = VAD()
    vad_stream = VADAudioStream(vad)
    samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "vad", "samples"))
    
    ww_setup = WakeWordSetup(
        vad, 
        samples_dir, 
        tts=container.tts, 
        rvc=container.rvc, 
        audio_player=player
    )
    if not ww_setup.has_enough_samples():
        tui_print("Recording Wakeword Samples...")
        ww_setup.new_wakeword_samples()
        
    wake_word = WakeWord(vad, samples_dir=samples_dir)
    
    QWEN3_LANG = container.config.get("QWEN3_LANG")
    USE_QWEN3_TTS = container.config.get("USE_QWEN3_TTS")
    lang = QWEN3_LANG if USE_QWEN3_TTS else "Spanish"

    try:
        while True:
            tui_print("\n[Client] Listening for activation word...")
            if not wake_word.listen_wakeword():
                continue
                
            tui_print("\n[Client] 📢 Wakeword detected!")
            
            # --- Acknowledgement Generated externally ---
            ack_prompt = get_acknowledgement_prompt(language=QWEN3_LANG)
            ack_text = clean_emojis(clean_think_tags(call_ollama(prompt="Generate heartbeat", system_prompt=ack_prompt, temperature=0.7)))
            tui_print(f"[🤖 ARIA] {ack_text}")
            
            wav, sr = container.tts.generate_speech(ack_text, languaje=lang)
            if wav is not None:
                if container.rvc:
                    wav, sr = container.rvc.transform_numpy(wav, sr)
                player.play(wav, sr)
            
            # Fluid Conversation Loop
            continuous = True
            # First interaction takes up to 10 seconds. Fluid replies wait a max of 4.
            timeout = 10.0 
            
            while continuous:
                tui_print("\n[Client] 🎤 Listening... ", end="", flush=True)
                vad_stream.start()
                vad_stream.clear_queue()
                audio_data = vad_stream.get_next_segment(timeout=timeout)
                vad_stream.stop()
                
                if audio_data is None:
                    tui_print("[Silence] Resuming Sleep Mode.")
                    break
                
                tui_print("[Speech Captured]. Processing...")
                
                initial_state = {
                    "input_audio": audio_data,
                    "user_text": "",
                    "history_context": "",
                    "tools_context": "",
                    "iteration_count": 0,
                    "rag_category": "none",
                    "rag_context": "",
                    "next_node": "asr",
                    "selected_category": "none",
                    "reply_text": "",
                    "start_time": time.time(),
                    "container": container
                }
                
                final_state = app.invoke(initial_state)
                
                if final_state.get("selected_category") == "exit":
                    continuous = False
                    tui_print("[System] 💤 Shutting down conversation...")
                    farewell_prompt = get_farewell_prompt(language=QWEN3_LANG)
                    farewell_text = clean_emojis(clean_think_tags(call_ollama(prompt="Generate farewell phrase", system_prompt=farewell_prompt, temperature=0.7)))
                    tui_print(f"[🤖 ARIA] {farewell_text}")
                    fw_wav, fw_sr = container.tts.generate_speech(farewell_text, languaje=lang)
                    if fw_wav is not None:
                        if container.rvc:
                            fw_wav, fw_sr = container.rvc.transform_numpy(fw_wav, fw_sr)
                        player.play(fw_wav, fw_sr)
                    sys.exit(0)
                
                tui_print(f"[🤖 ARIA] (Responding via TTS Stream...)")
                audio_generator = final_state.get("audio_stream")
                if audio_generator:
                    tui_play_stream(audio_generator, player)
                
                # Activate Continuous Listening
                timeout = 4.0 

    except KeyboardInterrupt:
        tui_print("\n[Client] 👋 Shutting down gracefully...")

if __name__ == "__main__":
    run_client()
