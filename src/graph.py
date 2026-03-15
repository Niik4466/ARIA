"""
Processing Graph Module (LangGraph).

Purpose: Define the voice processing pipeline using a decision and state-oriented 
architecture (Tool_Decisor -> Rag_Decisor -> Tool/Response).
"""

from typing import TypedDict, List, Literal
import re
import json
import time
import numpy as np
import os
import sys
from langgraph.graph import StateGraph, END

# Asegurar que el directorio raíz de ARIA esté en el PATH para todas las importaciones
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

from config import DECISOR_MODEL, RESPONSE_MODEL, USE_QWEN3_TTS, USE_RVC, QWEN3_LANG
from .vad.vad import VAD
from .vad.wakeword import WakeWord, WakeWordSetup
from .asr import ASR
from .agent import (
    call_ollama, 
    clean_think_tags, 
    clean_emojis,
    get_tool_decisor_prompt,
    get_rag_decisor_prompt,
    get_tool_agent_prompt,
    get_final_response_prompt,
    call_ollama_stream,
    get_acknowledgement_prompt,
    get_farewell_prompt,
    get_waiting_prompt
)
import threading
import queue
if USE_QWEN3_TTS:
    from .tts.qwen3_tts import Qwen3_TTS as Qwen3TTS
    from .tts.qwen3_tts import QwenVoiceSetup
else:
    from .tts.kokoro_tts import Kokoro_TTS as KokoroTTS
from .tts.rvc_backend import RVC_Backend
from .Tools.registry import execute_tool
from .rag import rag_manager
from .audio_io import player

# --- RAG Initialization ---
print("🔄 Initializing RAG system...")
RAG_CATEGORIES = rag_manager.update()
RAG_CATEGORIES_DESC_STR = "\n".join([f"- '{k}': {v}" for k, v in RAG_CATEGORIES.items()])
if not RAG_CATEGORIES_DESC_STR:
    RAG_CATEGORIES_DESC_STR = "(No documents available)"
print("✅ RAG Initialized.")


# --- Graph State ---
class GraphState(TypedDict, total=False):
    # Transcribed text from user
    user_text: str
    
    # State context
    history_context: str       # Accumulated chat text
    tools_context: str         # Accumulated text with tool results
    iteration_count: int       # Counter for loop protection
    
    # RAG results
    rag_category: str          # Selected RAG category
    rag_context: str           # Retrieved text

    # Flow control
    next_node: str             # 'tool_node', 'response_node' or 'end'
    selected_category: str     # 'search', 'os', 'basic', etc.
    
    # Final response
    reply_text: str
    
    # Metrics
    start_time: float

# --- Global Instances ---
vad = VAD()
asr = ASR(vad)
if USE_QWEN3_TTS:
    tts = Qwen3TTS()
    qwenVoiceSetup = QwenVoiceSetup(tts, asr)
else:
    tts = KokoroTTS()
rvc = None
if USE_RVC:
    rvc = RVC_Backend()
audio_player = player

import os
samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vad", "samples")

# Setup Wakeword if necessary
ww_setup = WakeWordSetup(vad, samples_dir, tts=tts, rvc=rvc, audio_player=audio_player)
if not ww_setup.has_enough_samples():
    ww_setup.new_wakeword_samples()

# Instantiate the listening wakeword
wake_word = WakeWord(vad, samples_dir=samples_dir)

# --- Nodes ---

def asr_node(state: GraphState) -> GraphState:
    """
    ASR NODE: Audio -> Text
    Converts audio to text and initializes history.
    """
    audio = asr.listen()
    text = asr.speech_to_text(audio) or ""
    print(f"[🎤 ASR] {text}")
    return {**state, "user_text": text}


def rag_decisor_node(state: GraphState) -> GraphState:
    """
    RAG DECISOR NODE: Defines if RAG is used and which category.
    Runs BEFORE Tool Decisor to provide context.
    """
    # Start timing for TTFS metric
    state["start_time"] = time.time()
    
    user_text = state["user_text"]
    
    # Skip if no categories
    if not RAG_CATEGORIES:
        return {**state, "rag_category": "none", "rag_context": ""}

    system_prompt = get_rag_decisor_prompt(RAG_CATEGORIES_DESC_STR)

    try:
        response = call_ollama(prompt=user_text, model=DECISOR_MODEL, system_prompt=system_prompt, think=False)
        # Clean response to get only the category
        selected = response.strip().replace("'", "").replace('"', '').replace(".", "").lower()
    except Exception as e:
        print(f"Error in RAG Decisor: {e}")
        selected = "none"

    rag_context = ""
    # Verify if category exists in our documents
    # Loose matching (lowercase)
    final_cat = "none"
    for cat in RAG_CATEGORIES.keys():
        if cat.lower() == selected:
            final_cat = cat
            break

    print(f"[📚 RAG Decisor] Selected: {final_cat}")

    if final_cat != "none":
        # Consult RAG
        print(f"[📚 RAG] Searching in '{final_cat}'...")
        rag_context = rag_manager.query_category(final_cat, user_text)
        if rag_context:
            print("[📚 RAG] Context retrieved.")
        else:
            print("[📚 RAG] No relevant information found.")
            
    return {
        **state,
        "rag_category": final_cat,
        "rag_context": rag_context
    }


def tool_decisor_node(state: GraphState) -> GraphState:
    """
    TOOL DECISOR NODE: Text + RAG -> Defines Agent Category
    """
    # Security: If max iterations reached, force response
    if state.get("iteration_count", 0) >= 5:
        print("[⚠️ Decisor] Max iterations reached. Forcing response.")
        return {**state, "next_node": "response"}

    user_text = state["user_text"]
    rag_context = state.get("rag_context", "")
    
    if not user_text:
        return {**state, "agent_category": "general"}
    
    # Inject RAG context into decisor prompt
    system_prompt = get_tool_decisor_prompt(
        tools_context=state.get("tools_context", ""),
        rag_context=rag_context, 
        history_context=state.get("history_context", "")
    )
    
    # Call Ollama
    try:
        response = call_ollama(prompt=user_text, model=DECISOR_MODEL, system_prompt=system_prompt)
        category = response.strip().lower()
        print(f"[🧠 Tool Decisor] said: {category}")
    except Exception as e:
        print(f"Error in Tool Decisor: {e}")
        category = "response"

    valid_categories = ['search', 'os', 'basic', 'autoconfig', 'exit']
    # Check if any valid category is in the response
    found_category = next((c for c in valid_categories if c in category), "response")
    
    if found_category == "exit":
        next_node = "end"
    else:
        next_node = "tool_node" if found_category != "response" else "response"
    
    return {
        **state,
        "next_node": next_node,
        "selected_category": found_category
    }


def tool_node(state: GraphState) -> GraphState:
    """
    Generates tool JSON and executes it.
    """
    category = state["selected_category"]
    
    # --- Quick Waiting Response ---
    if state.get("tools_context", "") == "" and category != "autoconfig":
        try:
            wait_prompt = get_waiting_prompt(language=QWEN3_LANG)
            wait_text = call_ollama(prompt="Generate waiting phrase", system_prompt=wait_prompt, temperature=0.7)
            wait_text = clean_think_tags(wait_text)
            wait_text = clean_emojis(wait_text)
            print(f"[🤖 ARIA] {wait_text}")
        
            lang = QWEN3_LANG if USE_QWEN3_TTS else "Spanish"
            wav, sr = tts.generate_speech(wait_text, languaje=lang)
        
            if wav is not None:
                if rvc:
                    wav, sr = rvc.transform_numpy(wav, sr)
                audio_player.play_async(wav, sr)
        except Exception as e:
            print(f"[System] ⚠️ Error in waiting response: {e}")

    history = state.get("history_context", "")
    user_text = state["user_text"]
    rag_context = state.get("rag_context", "")

    system_prompt = get_tool_agent_prompt(category=category, rag_context=rag_context, history_context=history)
    
    print(f"[🔧 ToolGen] Generating JSON for {category}...")
    response_json_str = call_ollama(
        prompt=user_text,
        model=RESPONSE_MODEL,
        system_prompt=system_prompt,
        json_mode=True,
    )
    
    # Exclude <think> tags from response
    response_json_str = clean_think_tags(response_json_str)
    
    tool_result_str = ""
    tool_name = "unknown"
    
    try:
        json_match = re.search(r'\{.*\}', response_json_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            tool_name = data.get("tool")
            tool_args = {k: v for k, v in data.items() if k != "tool"}
            
            print(f"[🔨 Executing] {tool_name} {tool_args}")
            res = execute_tool(tool_name, tool_args)
            
            tool_result_str = f"Tool '{tool_name}' executed. Result: {res}"
        else:
            tool_result_str = f"Error: No valid JSON generated. Response: {response_json_str}"
            
    except Exception as e:
        tool_result_str = f"Error executing tool: {e}"
    
    print(f"[✅ Result] {tool_result_str}")
    
    tools = state.get("tools_context", "")
    new_tools = tools + f"\n[Action] {tool_name} -> {tool_result_str}"
    
    return {
        **state,
        "tools_context": new_tools,
        "iteration_count": state["iteration_count"] + 1
    }


def integrated_response_node(state: GraphState) -> GraphState:
    """
    Fuses response generation and TTS into a single streaming pipeline.
    LLM (Stream) -> TTS (Sentence Stream) -> Audio Queue -> Playback Thread.
    """
    history = state.get("history_context", "")
    tools = state.get("tools_context", "")
    rag_context = state.get("rag_context", "")
    user_text = state["user_text"]
    
    print(f"[✨ Integrated] Retrieving conversation history from RAG...")
    rag_history = rag_manager.query_history(user_text, n_results=3)
    
    print(f"[✨ Integrated] Generating streaming response and audio...")

    # 1. Setup Audio Queue and Playback Worker
    audio_queue = queue.Queue()
    start_time = state.get("start_time", time.time())
    ttfs_measured = False
    
    def playback_worker():
        """Consumes audio chunks from the queue and plays them."""
        nonlocal ttfs_measured
        while True:
            item = audio_queue.get()
            if item is None:
                audio_queue.task_done()
                break
            
            # Measure TTFS on first audio chunk
            if not ttfs_measured:
                ttfs = time.time() - start_time
                print(f"\n[📊 METRIC] TIME TO FIRST SPEECH (TTFS): {ttfs:.4f} seconds")
                ttfs_measured = True

            wav, sr = item
            audio_player.play(wav, sr)
            audio_queue.task_done()

    pb_thread = threading.Thread(target=playback_worker, daemon=True)
    pb_thread.start()

    # 2. Text stream from Ollama
    lang = QWEN3_LANG if USE_QWEN3_TTS else "English"
    text_stream = call_ollama_stream(
        prompt=user_text, 
        model=RESPONSE_MODEL,
        system_prompt=get_final_response_prompt(tools_context=tools, history_context=rag_history, rag_context=rag_context, language=lang)
    )
    
    # 3. Wrap stream to capture full text for state
    full_text_list = []
    def text_spy(stream):
        for chunk in stream:
            full_text_list.append(chunk)
            yield chunk

    # 4. Generate audio from text stream and put in queue
    try:
        # We use the configured language or fallback to Spanish
        for wav, sr in tts.generate_speech_stream(text_spy(text_stream), languaje=lang):
            # Apply RVC if enabled
            if rvc:
                wav, sr = rvc.transform_numpy(wav, sr)
                
            audio_queue.put((wav, sr))
            
    except Exception as e:
        print(f"[Integrated Node] ❌ Error in stream processing: {e}")

    # 5. Cleanup: Signal worker to stop and wait
    audio_queue.put(None)
    audio_queue.join()
    pb_thread.join()
    
    full_reply = "".join(full_text_list)
    
    # Add historial to RAG
    print(f"[✨ Integrated] Registering conversation in RAG history...")
    rag_manager.add_to_history(user_text, full_reply)
    
    return {**state, "reply_text": full_reply}

# --- Graph ---

def route_decision(state: GraphState) -> Literal["tool_node", "response", "end"]:
    """Function for Conditional Edge."""
    return state["next_node"]


def build_graph():
    g = StateGraph(GraphState)
    
    g.add_node("asr", asr_node)
    g.add_node("rag_decisor", rag_decisor_node)
    g.add_node("tool_decisor", tool_decisor_node)
    g.add_node("tool_node", tool_node)
    g.add_node("response", integrated_response_node)
    
    g.set_entry_point("asr")
    
    # Flow: ASR -> RAG Decisor -> Tool Decisor
    g.add_edge("asr", "rag_decisor")
    g.add_edge("rag_decisor", "tool_decisor")
    
    # Decision -> Tool, Response or End
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

if __name__ == "__main__":
    run_aria()
