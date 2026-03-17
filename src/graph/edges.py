"""
Processing Graph Edges Module.

Purpose: Define the connections (edges) and final compilation of the LangGraph state machine.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
import sys
import os

# Ensure ARIA root is in PATH for all imports
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

from config import USE_QWEN3_TTS, QWEN3_LANG

# Import State and backend variables from state.py
from .state import (
    GraphState,
    wake_word,
    tts,
    rvc,
    audio_player
)

# Import Nodes from nodes.py
from .nodes import (
    asr_node,
    rag_decisor_node,
    tool_decisor_node,
    tool_node,
    integrated_response_node
)

from src.agent import (
    call_ollama,
    clean_think_tags,
    clean_emojis,
    get_acknowledgement_prompt,
    get_farewell_prompt
)

def route_decision(state: GraphState) -> Literal["tool_node", "response", "end"]:
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
    g.add_node("response", integrated_response_node)
    
    # Set the entry point of the pipeline
    g.set_entry_point("asr")
    
    # Flow: ASR -> RAG Decisor -> Tool Decisor
    g.add_edge("asr", "rag_decisor")
    g.add_edge("rag_decisor", "tool_decisor")
    
    # Conditional Decision -> Tool, Response or End
    g.add_conditional_edges(
        "tool_decisor",
        route_decision,
        {
            "tool_node": "tool_node",
            "response": "response",
            "end": END
        }
    )
    
    # Tool -> Back to Tool Decisor (Loop)
    g.add_edge("tool_node", "tool_decisor")
    
    # Response -> End
    g.add_edge("response", END)
    
    return g.compile()


def run_aria():
    """
    External loop: Wakeword -> Invoke Graph -> Wait for next wakeword.
    """
    app = build_graph()
    
    print("\n--- ARIA SYSTEM READY (External Loop) ---")
    print("Listening for activation word...")
    
    try:
        while True:
            # 1. Wait for wakeword
            if not wake_word.listen_wakeword():
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
                wav, sr = tts.generate_speech(ack_text, languaje=lang)
                
                if wav is not None:
                    if rvc:
                        wav, sr = rvc.transform_numpy(wav, sr)
                    audio_player.play(wav, sr)
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
                "start_time": 0
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
                    wav, sr = tts.generate_speech(farewell_text, languaje=lang)
                    
                    if wav is not None:
                        if rvc:
                            wav, sr = rvc.transform_numpy(wav, sr)
                        audio_player.play(wav, sr)
                except Exception as e:
                    print(f"[System] ⚠️ Error in farewell: {e}")
                
                break
            
            print("\n--- Returning to standby mode ---")
            print("Listening for activation word...")
            
    except KeyboardInterrupt:
        print("\n[System] 👋 Shutting down gracefully...")
