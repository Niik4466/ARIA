import pytest
import json
import os
import csv
import numpy as np
import sys
import time

# Ensure imports work from ARIA root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.container import Container
from src.graph.state import GraphState
from src.graph.nodes import (
    asr_node, 
    rag_decisor_node, 
    tool_decisor_node, 
    tool_node, 
    generate_response_node, 
    tts_response_node
)

NUM_ITERATIONS = 3

@pytest.fixture(scope="session")
def container():
    print("\n[Pytest] Initializing Component Container...")
    c = Container()
    
    # 1. Mute the audio player, calculate sleep time based on sr for realistic delay in gap simulation
    def fake_play(wav, sr):
        duration = len(wav) / sr
        time.sleep(duration / 2) # Speed up simulation while keeping an authentic streaming gap measurable
    
    c.audio_player.play = fake_play
    
    # 2. Mock MCP Manager to prevent hangs and simulate valid tools
    class MockTool:
        def __init__(self, t_id, desc, input_schema):
            self.id = t_id
            self.description = desc
            self.input_schema = input_schema
            self.name = t_id
            
    def mock_get_tools(*args, **kwargs):
        return [MockTool("weather_api.get_current_weather", "Get weather", {"type": "object", "properties": {"location": {"type": "string"}}})]
        
    def mock_execute_tool(name, args):
        time.sleep(1.0) # Simulate a server call latency
        return f'{{"success": true, "message": "Weather is sunny in {args.get("location", "Unknown")}"}}'
        
    c.mcp_manager.get_tools = mock_get_tools
    c.mcp_manager.execute_tool = mock_execute_tool
    
    return c

@pytest.fixture(scope="session")
def golden_dataset():
    path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(path, "r") as f:
        return json.load(f)

def run_graph_mocked(container, text_prompt):
    print(f"\n[Pytest] Generating mock audio for text: {text_prompt}")
    
    # 1. Provide mock audio using TTS
    try:
        # Some TTS modules return wav, sr directly. Using the first chunk if it streams.
        lang = "English"
        generator_or_tuple = container.tts.generate_speech(text_prompt, languaje=lang)
        if isinstance(generator_or_tuple, tuple):
            wav, sr = generator_or_tuple
        else:
            wav, sr = next(generator_or_tuple) # Handle stream if generate_speech is generator somehow
    except Exception as e:
        print(f"[Pytest] Warning - Could not mock via actual TTS ({e}). Using empty array.")
        wav, sr = np.zeros(16000, dtype=np.float32), 16000
    
    # 2. Patch listner
    original_listen = getattr(container.asr, "listen", lambda: None)
    container.asr.listen = lambda timeout=10.0: wav
    
    # 3. Simulate Graph State iteration
    state: GraphState = {
        "container": container,
        "history_context": "",
        "tools_context": "",
        "iteration_count": 0,
        "performance_metrics": []
    }
    
    try:
        state = asr_node(state)
        state = rag_decisor_node(state)
        
        while state.get("next_node") not in ["generate_response", "end"]:
            if state["next_node"] == "tool_decisor":
                state = tool_decisor_node(state)
            elif state["next_node"] == "tool_node":
                state = tool_node(state)
                # Safeguard
                if state.get("iteration_count", 0) > 3:
                    state["next_node"] = "generate_response"
                    break
                
        if state.get("next_node") != "end":
            state = generate_response_node(state)
            state = tts_response_node(state)
            
    finally:
        container.asr.listen = original_listen
        
    return state.get("performance_metrics", [])

def test_performance(container, golden_dataset):
    results = []
    
    for item in golden_dataset:
        prompt_id = item["id"]
        category = item.get("category", "Uncategorized")
        text_prompt = item["prompt"]
        
        # Accumulate across N iterations
        all_metrics = {}
        
        for i in range(NUM_ITERATIONS):
            print(f"\n--- Running iteration {i + 1}/{NUM_ITERATIONS} for [{category}] {prompt_id} ---")
            metrics = run_graph_mocked(container, text_prompt)
            for m in metrics:
                op = m["operation"]
                duration = m["duration"]
                if op not in all_metrics:
                    all_metrics[op] = []
                all_metrics[op].append(duration)
                
        for op, durations in all_metrics.items():
            avg_time = float(np.mean(durations))
            std_time = float(np.std(durations)) if len(durations) > 1 else 0.0
            results.append({
                "category": category,
                "prompt_id": prompt_id,
                "operation": op,
                "mean_time": avg_time,
                "std_time": std_time
            })
            
    output_path = os.path.join(os.path.dirname(__file__), "performance_results.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "prompt_id", "operation", "mean_time", "std_time"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n✅ All iterations completed. Results compiled into {output_path}")
    assert len(results) > 0
