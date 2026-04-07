"""
Processing Graph State Module.

Purpose: Define the GraphState type definition.
"""

from typing import TypedDict, Any
import os
import sys

# Ensure ARIA root is in PATH for all imports
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

from src.container import Container

# --- Graph State ---
class GraphState(TypedDict, total=False):
    # Input audio
    input_audio: Any
    
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
    next_node: str             # 'tool_node', 'generate_response' or 'end'
    selected_category: str     # 'search', 'os', 'basic', etc.
    
    # Stream capability
    text_stream: Any           # Generator for LLM's text output
    audio_stream: Any          # Generator for TTS output
    
    # Final response
    reply_text: str
    
    # Metrics
    start_time: float
    performance_metrics: list

    # Backend Container
    container: Container
