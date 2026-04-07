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


