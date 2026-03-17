"""
Processing Graph Nodes Module.

Purpose: Define the states (nodes) for the voice processing pipeline 
(ASR, RAG Decisor, Tool Decisor, Tool Gen, Integrated Response).
"""

from typing import TypedDict, List, Literal
import re
import json
import time
import numpy as np
import os
import sys
import threading
import queue

# Ensure ARIA root is in PATH for all imports
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

from ..utils import Config

config = Config()

DECISOR_MODEL = config.get("DECISOR_MODEL")
RESPONSE_MODEL = config.get("RESPONSE_MODEL")
USE_QWEN3_TTS = config.get("USE_QWEN3_TTS")
USE_RVC = config.get("USE_RVC")
QWEN3_LANG = config.get("QWEN3_LANG")
from src.vad.vad import VAD
from src.vad.wakeword import WakeWord, WakeWordSetup
from src.asr import ASR
from src.agent import (
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
from .state import (
    GraphState,
    RAG_CATEGORIES,
    RAG_CATEGORIES_DESC_STR,
    asr,
    tts,
    rvc,
    audio_player
)
from src.Tools.registry import execute_tool
from src.rag import rag_manager

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
    
    # Add history to RAG
    print(f"[✨ Integrated] Registering conversation in RAG history...")
    rag_manager.add_to_history(user_text, full_reply)
    
    return {**state, "reply_text": full_reply}
