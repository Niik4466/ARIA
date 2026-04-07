import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.graph.state import GraphState
from src.graph.edges import build_graph

# We intercept nodes to verify whether they executed by logging to state.

@pytest.fixture
def mock_graph_container():
    """Mock container dependencies heavily so the graph logic operates instantaneously."""
    container_instance = MagicMock()
    
    # Audio Player Mocks
    container_instance.audio_player = MagicMock()
    
    # ASR Mock
    container_instance.asr = MagicMock()
    container_instance.asr.speech_to_text.return_value = "hello"
    
    # TTS Mock
    container_instance.tts = MagicMock()
    container_instance.tts.generate_speech.return_value = (None, None)
    
    def mock_speech_stream(stream, **kwargs):
        for chunk in stream:
            pass # Exhaust the generator to trigger text_spy
        yield (None, None)
        
    container_instance.tts.generate_speech_stream.side_effect = mock_speech_stream
    
    # MCP Manager Mock
    container_instance.mcp_manager = MagicMock()
    container_instance.mcp_manager.get_tools.return_value = []
    container_instance.mcp_manager.execute_tool.return_value = {"result": "tool_executed"}
    
    # RAG Manager Mock
    container_instance.rag_manager = MagicMock()
    container_instance.rag_manager.query_documents.return_value = "Found RAG context."
    container_instance.rag_manager.query_history.return_value = "Found RAG history."
    
    # Memory Manager Mock
    container_instance.memory_manager = MagicMock()
    container_instance.memory_manager.retrieve_relevant_insights.return_value = ""
    container_instance.memory_manager.handle_dual_query.return_value = None
    
    return container_instance

def test_graph_route_base_response(mock_graph_container):
    """
    Test the base routing: asr_node -> rag_decisor -> tool_decisor -> generate -> tts -> END
    """
    mock_graph_container.rag_manager.query_documents.return_value = ""
    
    # Mock LLM decision path: NO RAG, NO TOOL (Response mode)
    with patch("src.graph.nodes.call_ollama") as mock_ollama, \
         patch("src.graph.nodes.call_ollama_stream") as mock_generate:
        
        # Only Tool ("response") evaluates because RAG logic returns early due to empty mock context
        mock_ollama.side_effect = ["response"]
        
        # Stream logic mock
        mock_generate.return_value = ["Hi ", "there"]
        
        app = build_graph()
        initial_state = GraphState(
            performance_metrics=[],
            container=mock_graph_container,
            user_text="hello",
            history_context="",
            tools_context="",
            iteration_count=0,
            rag_category="none",
            rag_context="",
            next_node="asr",
            selected_category="none",
            reply_text="",
            response="",
            start_time=0
        )
        
        # Execute End-To-End Graph
        final_state = app.invoke(initial_state)
        
        # Assertions to ensure edges routed correctly
        assert final_state["user_text"].lower() == "hello", "ASR node did not inject input."
        assert not final_state.get("needs_rag", False), "RAG decisor should evaluate NO."
        assert final_state.get("selected_category", "") == "response", "Tool decisor should evaluate RESPONSE."
        
        # Context arrays shouldn't exist / be populated
        assert not final_state.get("rag_context"), "RAG logic bypassed, context should be empty."
        assert not final_state.get("tools_context"), "Tool execution bypassed, result should be empty."
        
        # Final response validation
        assert "Hi there" in final_state["reply_text"], "Generate_response node did not construct LLM buffer correctly."


def test_graph_route_rag_injection(mock_graph_container):
    """
    Test RAG routing correctly inserts context into final generation:
    asr_node -> rag_decisor -> rag_node -> generate -> tts -> END
    """
    with patch("src.graph.nodes.call_ollama") as mock_ollama, \
         patch("src.graph.nodes.call_ollama_stream") as mock_generate:
        
        # Return "yes" for RAG decisor. Graph routing should bypass 'tool_decisor' and go Straight to generation via rag_node!
        mock_ollama.return_value = "yes"
        mock_generate.return_value = ["Based ", "on ", "RAG..."]
        
        app = build_graph()
        initial_state = GraphState(
            performance_metrics=[],
            container=mock_graph_container,
            user_text="hello",
            rag_category="none",
            next_node="asr",
            selected_category="none"
        )
        final_state = app.invoke(initial_state)
        
        # "needs_rag" is technically not used, it sets next_node="generate_response"
        assert final_state.get("next_node") == "generate_response"
        assert final_state.get("rag_context") == "REFERENCE DOCUMENTS:\nFound RAG context.", "rag_node failed to execute or inject array."
        assert "Based on RAG..." in final_state["reply_text"], "Response missing stream generation."


from src.graph.nodes import VALID_CATEGORIES

@pytest.mark.parametrize("category", VALID_CATEGORIES)
def test_graph_route_dynamic_tool_decisor(mock_graph_container, category):
    """
    Dynamically tests the routing for each valid category inside the tool decisor natively.
    """
    with patch("src.graph.nodes.call_ollama") as mock_ollama, \
         patch("src.graph.nodes.call_ollama_stream") as mock_generate:
        
        if category == "tool":
            mock_ollama.side_effect = ["no", "tool", "Generating tool schema...", '{"tool": "fake_tool", "arguments": {}}']
        else:
            mock_ollama.side_effect = ["no", category]
            
        mock_generate.return_value = ["Dynamic ", "run."]
        
        app = build_graph()
        initial_state = GraphState(
            performance_metrics=[],
            container=mock_graph_container,
            user_text="dynamic test",
            history_context="",
            tools_context="",
            iteration_count=0,
            rag_category="none",
            rag_context="",
            next_node="asr",
            selected_category="none",
            reply_text="",
            response="",
            start_time=0
        )
        final_state = app.invoke(initial_state)
        
        if category == "exit":
            assert final_state.get("selected_category") == "exit"
            assert final_state.get("next_node") == "end"
            assert not final_state.get("reply_text")
        elif category == "tool":
            # the tool loop converges to response natively
            assert final_state.get("selected_category") == "response"
            assert "tool_executed" in str(final_state.get("tools_context", ""))
            assert "Dynamic run." in final_state["reply_text"]
        elif category == "response":
            assert final_state.get("selected_category") == "response"
            assert "Dynamic run." in final_state["reply_text"]

def test_langgraph_structural_edges():
    """LangGraph built-in structure introspection."""
    app = build_graph()
    graph_map = app.get_graph()
    
    # Assert structural integrity directly independent of invoke.
    nodes = list(graph_map.nodes.keys())
    assert "asr" in nodes
    assert "rag_decisor" in nodes
    assert "tool_decisor" in nodes
    assert "generate_response" in nodes
