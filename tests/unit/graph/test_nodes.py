import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from unittest.mock import patch, MagicMock

def get_base_state():
    mock_container = MagicMock()
    # Provide mock returns for heavily used things
    mock_container.rag_manager.query_documents.return_value = "Fake Context"
    mock_container.memory_manager.retrieve_relevant_insights.return_value = ""
    mock_container.memory_manager.handle_dual_query.return_value = None
    return {
        "user_text": "Hi",
        "container": mock_container,
        "iteration_count": 0
    }

def test_rag_decisor_node_yes():
    from src.graph.nodes import rag_decisor_node
    
    with patch("src.graph.nodes.call_ollama", return_value="Yes"):
        state = get_base_state()
        new_state = rag_decisor_node(state)
        
        assert new_state["next_node"] == "generate_response"
        assert new_state["rag_context"] == "REFERENCE DOCUMENTS:\nFake Context"

def test_rag_decisor_node_no():
    from src.graph.nodes import rag_decisor_node
    
    with patch("src.graph.nodes.call_ollama", return_value="No"):
        state = get_base_state()
        new_state = rag_decisor_node(state)
        
        assert new_state["next_node"] == "tool_decisor"

def test_tool_decisor_node_tool():
    from src.graph.nodes import tool_decisor_node
    
    with patch("src.graph.nodes.call_ollama", return_value="tool"):
        state = get_base_state()
        new_state = tool_decisor_node(state)
        
        assert new_state["next_node"] == "tool_node"
        assert new_state["selected_category"] == "tool"

def test_tool_node_json_parsing():
    from src.graph.nodes import tool_node
    
    # Return strict json mock
    with patch("src.graph.nodes.call_ollama", return_value='{"tool": "weather_api", "location": "Paris"}'):
        state = get_base_state()
        
        # Mock what tool gets executed
        state["container"].mcp_manager.execute_tool.return_value = "Sunny"
        
        new_state = tool_node(state)
        
        assert "weather_api" in new_state["tools_context"]
        assert "Sunny" in new_state["tools_context"]
        assert new_state["iteration_count"] == 1

def test_generate_response_node():
    from src.graph.nodes import generate_response_node
    
    with patch("src.graph.nodes.call_ollama_stream", return_value=iter(["Hello ", "world"])):
        state = get_base_state()
        new_state = generate_response_node(state)
        
        assert new_state["text_stream"] is not None
        assert list(new_state["text_stream"]) == ["Hello ", "world"]
