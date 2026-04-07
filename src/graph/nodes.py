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
    get_waiting_prompt
)
from .state import GraphState

# Tool decisor valid categories decisions
VALID_CATEGORIES = ['tool', 'response', 'exit']

# --- Nodes ---

def asr_node(state: GraphState) -> GraphState:
    """
    ASR NODE: Audio -> Text
    Converts audio to text and initializes history.
    """
    state.setdefault("performance_metrics", [])
    t0 = time.time()
    container = state["container"]
    audio = container.asr.listen()
    t1_listen = time.time()
    state["performance_metrics"].append({"operation": "asr_listen", "duration": t1_listen - t0})
    
    t1 = time.time()
    text = container.asr.speech_to_text(audio) or ""
    t2 = time.time()
    state["performance_metrics"].append({"operation": "asr_speech_to_text", "duration": t2 - t1})
    
    print(f"[🎤 ASR] {text}")
    return {**state, "user_text": text}


def rag_decisor_node(state: GraphState) -> GraphState:
    """
    RAG DECISOR NODE: Searches the RAG and decides if the context 
    is sufficient to answer the user.
    """
    state["start_time"] = time.time()
    state.setdefault("performance_metrics", [])
    user_text = state["user_text"]
    container = state["container"]
    
    # Check Extended STM for Dual Query
    q2 = container.memory_manager.handle_dual_query(user_text)
    
    print(f"[📚 RAG] Querying Information for: '{user_text}'...")
    t_start_rag = time.time()
    rag_context = container.rag_manager.query_documents(user_text, query2=q2)
    state["performance_metrics"].append({"operation": "rag_query_documents", "duration": time.time() - t_start_rag})
    
    # Retrieve LTM Insights and inject
    insights = container.memory_manager.retrieve_relevant_insights(user_text)
    
    # Combine Contexts
    full_context = ""
    if insights:
        full_context += f"USER INSIGHTS (Traits/Preferences):\n{insights}\n\n"
    if rag_context:
        full_context += f"REFERENCE DOCUMENTS:\n{rag_context}\n"
        
    full_context = full_context.strip()

    if not full_context:
        print("[📚 RAG Decisor] No relevant context or insights found. Proceeding to tools.")
        return {**state, "rag_context": "", "next_node": "tool_decisor"}

    print("[📚 RAG Decisor] Context retrieved. Evaluating relevance...")
    system_prompt = get_rag_decisor_prompt(full_context)

    try:
        t_start_llm = time.time()
        response = call_ollama(prompt=user_text, model=DECISOR_MODEL, system_prompt=system_prompt, think=False)
        state["performance_metrics"].append({"operation": "rag_decisor_llm", "duration": time.time() - t_start_llm})
        decision = response.strip().lower().replace("'", "").replace('"', '').replace(".", "")
    except Exception as e:
        print(f"Error in RAG Decisor: {e}")
        decision = "no"

    print(f"[📚 RAG Decisor] Decision: {decision}")

    if "yes" in decision:
        print("[🧠 RAG Decisor] Context is sufficient. Routing to response.")
        return {**state, "rag_context": full_context, "next_node": "generate_response"}
    else:
        print("[🧠 RAG Decisor] Context insufficient. Proceeding to tools.")
        return {**state, "rag_context": full_context, "next_node": "tool_decisor"}


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
        t_start_llm = time.time()
        response = call_ollama(prompt=user_text, model=DECISOR_MODEL, system_prompt=system_prompt)
        state.setdefault("performance_metrics", []).append({"operation": "tool_decisor_llm", "duration": time.time() - t_start_llm})
        category = response.strip().lower()
        print(f"[🧠 Tool Decisor] said: {category}")
    except Exception as e:
        print(f"Error in Tool Decisor: {e}")
        category = "response"

    valid_categories = VALID_CATEGORIES
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
    t_start_llm = time.time()
    response_json_str = call_ollama(
        prompt=user_text,
        model=RESPONSE_MODEL,
        system_prompt=system_prompt,
        json_mode=True,
    )
    state.setdefault("performance_metrics", []).append({"operation": "tool_node_llm", "duration": time.time() - t_start_llm})
    
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
            t_start_exec = time.time()
            res = container.mcp_manager.execute_tool(tool_name, tool_args)
            state.setdefault("performance_metrics", []).append({"operation": f"tool_eval_{tool_name}", "duration": time.time() - t_start_exec})
            
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
    t_start_llm = time.time()
    state.setdefault("performance_metrics", []).append({"operation": "generate_response_llm_init", "duration": time.time() - t_start_llm})
    
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
    
    playback_gaps = []
    last_play_time = None
    
    def playback_worker():
        nonlocal ttfs_measured, last_play_time
        while True:
            item = audio_queue.get()
            if item is None:
                audio_queue.task_done()
                break
            
            current_time = time.time()
            if last_play_time is not None:
                playback_gaps.append(current_time - last_play_time)
            
            # Measure TTFS on first audio chunk
            if not ttfs_measured:
                ttfs = current_time - start_time
                state.setdefault("performance_metrics", []).append({"operation": "ttfs", "duration": ttfs})
                print(f"\n[📊 METRIC] TIME TO FIRST SPEECH (TTFS): {ttfs:.4f} seconds")
                ttfs_measured = True

            wav, sr = item
            container.audio_player.play(wav, sr)
            last_play_time = time.time()
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
    
    # Calculate streaming gap delay metrics
    if playback_gaps:
        avg_gap = sum(playback_gaps) / len(playback_gaps)
        state.setdefault("performance_metrics", []).append({"operation": "tts_streaming_gap_avg", "duration": avg_gap})
        std_gap = float(np.std(playback_gaps))
        state.setdefault("performance_metrics", []).append({"operation": "tts_streaming_gap_std", "duration": std_gap})
    
    full_reply = "".join(full_text_list)
    
    # Update STM and LTM Memory Systems
    print(f"[✨ Integrated] Updating Memory System...")
    container.memory_manager.update_after_interaction(user_text, full_reply)
    
    return {**state, "reply_text": full_reply}
