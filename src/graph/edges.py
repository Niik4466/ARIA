"""
Processing Graph Edges Module.

Purpose: Define the connections (edges) and final compilation of the LangGraph state machine.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from ..utils import Config
import sys
import os

# Ensure ARIA root is in PATH for all imports
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

config = Config()

USE_QWEN3_TTS = config.get("USE_QWEN3_TTS")
QWEN3_LANG = config.get("QWEN3_LANG")

# Import State and backend variables from state.py
from .state import GraphState

from .nodes import (
    asr_node,
    rag_decisor_node,
    tool_decisor_node,
    tool_node,
    generate_response_node,
    tts_response_node
)

from src.agent import (
    call_ollama,
    clean_think_tags,
    clean_emojis,
    get_acknowledgement_prompt,
    get_farewell_prompt
)

def route_decision(state: GraphState) -> Literal["tool_node", "generate_response", "end", "tool_decisor"]:
    """Function for Conditional Edge."""
    return state["next_node"]


def build_graph():
    """Builds and compiles the processing graph."""
    g = StateGraph(GraphState)
    
    # Add nodes to the graph
    g.add_node("asr", asr_node)
    g.add_node("rag_decisor", rag_decisor_node)
    g.add_node("tool_decisor", tool_decisor_node)
    g.add_node("tool_node", tool_node)
    g.add_node("generate_response", generate_response_node)
    g.add_node("tts_response", tts_response_node)
    
    # Set the entry point of the pipeline
    g.set_entry_point("asr")
    
    # Flow: ASR -> RAG Decisor
    g.add_edge("asr", "rag_decisor")
    
    # RAG Decisor routes to either Response (if RAG context is enough) or Tool Decisor
    g.add_conditional_edges(
        "rag_decisor",
        route_decision,
        {
            "tool_decisor": "tool_decisor",
            "generate_response": "generate_response"
        }
    )
    
    # Conditional Decision -> Tool, Generate Response or End
    g.add_conditional_edges(
        "tool_decisor",
        route_decision,
        {
            "tool_node": "tool_node",
            "generate_response": "generate_response",
            "end": END
        }
    )
    
    # Tool -> Back to Tool Decisor (Loop)
    g.add_edge("tool_node", "tool_decisor")
    
    # Generate Response -> TTS Response
    g.add_edge("generate_response", "tts_response")
    
    # TTS Response -> End
    g.add_edge("tts_response", END)
    
    return g.compile()


def run_aria(container):
    """
    External loop: Wakeword -> Invoke Graph -> Wait for next wakeword.
    """
    app = build_graph()
    
    print("\n--- ARIA SYSTEM READY (External Loop) ---")
    print("Listening for activation word...")
    
    try:
        while True:
            # 1. Wait for wakeword
            if not container.wake_word.listen_wakeword():
                continue
            
            print("[System] 📢 Wakeword detected! Starting dialogue...")
            
            # --- Acknowledgement Response ---
            try:
                # Generate text using agent
                ack_prompt = get_acknowledgement_prompt(language=QWEN3_LANG)
                ack_text = call_ollama(prompt="Generate heartbeat/activation phrase", system_prompt=ack_prompt, temperature=0.7)
                ack_text = clean_think_tags(ack_text)
                ack_text = clean_emojis(ack_text)
                print(f"[🤖 ARIA] {ack_text}")
                
                # Generate and play audio
                lang = QWEN3_LANG if USE_QWEN3_TTS else "Spanish"
                wav, sr = container.tts.generate_speech(ack_text, languaje=lang)
                
                if wav is not None:
                    if container.rvc:
                        wav, sr = container.rvc.transform_numpy(wav, sr)
                    container.audio_player.play(wav, sr)
            except Exception as e:
                print(f"[System] ⚠️ Error in acknowledgement: {e}")
            
            # 2. Initialize state
            initial_state = {
                "user_text": "",
                "history_context": "",
                "tools_context": "",
                "iteration_count": 0,
                "rag_category": "none",
                "rag_context": "",
                "next_node": "asr",
                "selected_category": "none",
                "reply_text": "",
                "start_time": 0,
                "container": container
            }
            
            # 3. Invoke graph once
            final_state = app.invoke(initial_state)
            
            # 4. Check if we should exit the program
            if final_state.get("selected_category") == "exit":
                print("[System] 💤 Exit command received. Shutting down...")
                
                # --- Farewell Response ---
                try:
                    farewell_prompt = get_farewell_prompt(language=QWEN3_LANG)
                    farewell_text = call_ollama(prompt="Generate farewell phrase", system_prompt=farewell_prompt, temperature=0.7)
                    farewell_text = clean_think_tags(farewell_text)
                    farewell_text = clean_emojis(farewell_text)
                    print(f"[🤖 ARIA] {farewell_text}")
                    
                    lang = QWEN3_LANG if USE_QWEN3_TTS else "Spanish"
                    wav, sr = container.tts.generate_speech(farewell_text, languaje=lang)
                    
                    if wav is not None:
                        if container.rvc:
                            wav, sr = container.rvc.transform_numpy(wav, sr)
                        container.audio_player.play(wav, sr)
                except Exception as e:
                    print(f"[System] ⚠️ Error in farewell: {e}")
                
                break
            
            print("\n--- Returning to standby mode ---")
            print("Listening for activation word...")
            
    except KeyboardInterrupt:
        print("\n[System] 👋 Shutting down gracefully...")
