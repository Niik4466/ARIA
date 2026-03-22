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
from .state import GraphState

# --- Nodes ---

def asr_node(state: GraphState) -> GraphState:
    """
    ASR NODE: Audio -> Text
    Converts audio to text and initializes history.
    """
    container = state["container"]
    audio = container.asr.listen()
    text = container.asr.speech_to_text(audio) or ""
    print(f"[🎤 ASR] {text}")
    return {**state, "user_text": text}


def rag_decisor_node(state: GraphState) -> GraphState:
    """
    RAG DECISOR NODE: Searches the RAG and decides if the context 
    is sufficient to answer the user.
    """
    state["start_time"] = time.time()
    user_text = state["user_text"]
    container = state["container"]
    
    print(f"[📚 RAG] Querying Information for: '{user_text}'...")
    rag_context = container.rag_manager.query_documents(user_text)
    
    if not rag_context or not rag_context.strip():
        print("[📚 RAG Decisor] No relevant context found. Proceeding to tools.")
        return {**state, "rag_context": "", "next_node": "tool_decisor"}

    print("[📚 RAG Decisor] Context retrieved. Evaluating relevance...")
    system_prompt = get_rag_decisor_prompt(rag_context)

    try:
        response = call_ollama(prompt=user_text, model=DECISOR_MODEL, system_prompt=system_prompt, think=False)
        decision = response.strip().lower().replace("'", "").replace('"', '').replace(".", "")
    except Exception as e:
        print(f"Error in RAG Decisor: {e}")
        decision = "no"

    print(f"[📚 RAG Decisor] Decision: {decision}")

    if "yes" in decision:
        print("[🧠 RAG Decisor] Context is sufficient. Routing to response.")
        return {**state, "rag_context": rag_context, "next_node": "generate_response"}
    else:
        print("[🧠 RAG Decisor] Context insufficient. Proceeding to tools.")
        return {**state, "next_node": "tool_decisor"}


def tool_decisor_node(state: GraphState) -> GraphState:
    """
    TOOL DECISOR NODE: Text + RAG -> Defines Agent Category
    """
    # Security: If max iterations reached, force response
    if state.get("iteration_count", 0) >= 5:
        print("[⚠️ Decisor] Max iterations reached. Forcing response.")
        return {**state, "next_node": "generate_response"}

    user_text = state["user_text"]
    rag_context = state.get("rag_context", "")
    
    if not user_text:
        return {**state, "agent_category": "general"}
    
    # Inject RAG context into decisor prompt
    system_prompt = get_tool_decisor_prompt(
        tools_context=state.get("tools_context", ""),
    )
    
    # Call Ollama
    try:
        response = call_ollama(prompt=user_text, model=DECISOR_MODEL, system_prompt=system_prompt)
        category = response.strip().lower()
        print(f"[🧠 Tool Decisor] said: {category}")
    except Exception as e:
        print(f"Error in Tool Decisor: {e}")
        category = "response"

    valid_categories = ['tool', 'response', 'exit']
    found_category = next((c for c in valid_categories if c in category), "response")
    
    if found_category == "exit":
        next_node = "end"
    else:
        next_node = "tool_node" if found_category == "tool" else "generate_response"
    
    return {
        **state,
        "next_node": next_node,
        "selected_category": found_category
    }


def tool_node(state: GraphState) -> GraphState:
    """
    Generates tool JSON and executes it dynamically using MCP tools.
    """
    container = state["container"]
    
    # --- Quick Waiting Response ---
    if state.get("tools_context", "") == "":
        try:
            wait_prompt = get_waiting_prompt(language=QWEN3_LANG)
            wait_text = call_ollama(prompt="Generate waiting phrase", system_prompt=wait_prompt, temperature=0.7)
            wait_text = clean_think_tags(wait_text)
            wait_text = clean_emojis(wait_text)
            print(f"[🤖 ARIA] {wait_text}")
        
            lang = QWEN3_LANG if USE_QWEN3_TTS else "Spanish"
            wav, sr = container.tts.generate_speech(wait_text, languaje=lang)
        
            if wav is not None:
                if container.rvc:
                    wav, sr = container.rvc.transform_numpy(wav, sr)
                container.audio_player.play_async(wav, sr)
        except Exception as e:
            print(f"[System] ⚠️ Error in waiting response: {e}")

    history = state.get("history_context", "")
    user_text = state["user_text"]
    rag_context = state.get("rag_context", "")
    
    # Fetch top 3 tools directly from MCP
    mcp_tools = container.mcp_manager.get_tools(user_text, k=3)
    if not mcp_tools:
        tools_desc = "No tools available."
    else:
        tools_desc = "\n".join([f"- ID: {t.id} | Desc: {t.description}\n  Schema: {json.dumps(t.input_schema)}" for t in mcp_tools])

    system_prompt = get_tool_agent_prompt(tools_desc=tools_desc, rag_context=rag_context, history_context=history)
    
    print(f"[🔧 ToolGen] Generating JSON...")
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
            res = container.mcp_manager.execute_tool(tool_name, tool_args)
            
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


def generate_response_node(state: GraphState) -> GraphState:
    """
    GENERATE RESPONSE NODE:
    Generates the text response via LLM streaming.
    Passes the stream to the TTS node.
    """
    container = state["container"]
    history = state.get("history_context", "")
    tools = state.get("tools_context", "")
    rag_context = state.get("rag_context", "")
    user_text = state["user_text"]
    
    print(f"[✨ Generate] Retrieving conversation history from RAG...")
    rag_history = container.rag_manager.query_history(user_text, n_results=3)
    
    print(f"[✨ Generate] Generating streaming response...")

    # Text stream from Ollama
    lang = QWEN3_LANG if USE_QWEN3_TTS else "English"
    text_stream = call_ollama_stream(
        prompt=user_text, 
        model=RESPONSE_MODEL,
        system_prompt=get_final_response_prompt(tools_context=tools, history_context=rag_history, rag_context=rag_context, language=lang)
    )
    
    return {**state, "text_stream": text_stream}


def tts_response_node(state: GraphState) -> GraphState:
    """
    TTS RESPONSE NODE:
    Consumes the text stream, generating audio chunks and playing them.
    Saves the final text to the RAG history context.
    """
    container = state["container"]
    text_stream = state.get("text_stream")
    user_text = state.get("user_text", "")
    lang = QWEN3_LANG if USE_QWEN3_TTS else "English"
    
    if not text_stream:
        print("[System] ⚠️ Error: No text stream provided to TTS node.")
        return state
        
    print(f"[🔊 TTS] Generating audio from stream...")

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
            container.audio_player.play(wav, sr)
            audio_queue.task_done()

    pb_thread = threading.Thread(target=playback_worker, daemon=True)
    pb_thread.start()

    # 2. Wrap stream to capture full text for state
    full_text_list = []
    def text_spy(stream):
        for chunk in stream:
            full_text_list.append(chunk)
            yield chunk

    # 3. Generate audio from text stream and put in queue
    try:
        # We use the configured language or fallback to Spanish
        for wav, sr in container.tts.generate_speech_stream(text_spy(text_stream), languaje=lang):
            # Apply RVC if enabled
            if container.rvc:
                wav, sr = container.rvc.transform_numpy(wav, sr)
                
            audio_queue.put((wav, sr))
            
    except Exception as e:
        print(f"[TTS Node] ❌ Error in stream processing: {e}")

    # 4. Cleanup: Signal worker to stop and wait
    audio_queue.put(None)
    audio_queue.join()
    pb_thread.join()
    
    full_reply = "".join(full_text_list)
    
    # Add history to RAG
    print(f"[✨ Integrated] Registering conversation in RAG history...")
    container.rag_manager.add_to_history(user_text, full_reply)
    
    return {**state, "reply_text": full_reply}
