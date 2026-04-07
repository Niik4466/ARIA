import pytest
import time
from unittest.mock import MagicMock, patch
from collections import deque
from src.memory import MemoryManager

@pytest.fixture
def mock_rag_manager():
    rag = MagicMock()
    # default mock return
    rag.get_insights.return_value = [
        {"text": "user is beginner", "similarity": 0.4},
        {"text": "user noise", "similarity": 0.1}
    ]
    return rag


@pytest.fixture
def memory_manager(mock_rag_manager):
    # Initialize MemoryManager with dummy config
    def mock_config_get(key, default=None):
        if key == "DECISOR_MODEL": return "dummy-model"
        if key == "IMMEDIATE_STM_TTL": return 30.0
        if key == "EXTENDED_STM_TTL": return 120.0
        if key == "INSIGHT_THRESHOLD": return 0.35
        return default

    with patch("src.memory.Config.get", side_effect=mock_config_get):
        return MemoryManager(mock_rag_manager)


def test_immediate_stm_ttl(memory_manager):
    # Setup state
    memory_manager.immediate_stm.append({"user": "hello", "agent": "hi"})
    memory_manager.last_interaction_time = time.time() - 40 # 40 seconds ago
    
    # Trigger cleanup
    memory_manager._cleanup_stm()
    
    # Should be empty due to 30s TTL
    assert len(memory_manager.immediate_stm) == 0


def test_extended_stm_ttl(memory_manager):
    memory_manager.extended_stm = "intent: learn"
    memory_manager.extended_stm_time = time.time() - 130 # 130 seconds ago
    
    memory_manager._cleanup_stm()
    assert memory_manager.extended_stm is None


def test_retrieve_relevant_insights_threshold(memory_manager, mock_rag_manager):
    # insight_threshold is default 0.35
    insights = memory_manager.retrieve_relevant_insights("some query")
    
    # Ensure it only returns the one above 0.35
    assert "- user is beginner" in insights
    assert "user noise" not in insights


@patch("src.memory.call_ollama")
def test_handle_dual_query_valid(mock_ollama, memory_manager):
    mock_ollama.return_value = "enhanced Q2"
    memory_manager.extended_stm = "intent: learn"
    memory_manager.extended_stm_time = time.time()
    
    q2 = memory_manager.handle_dual_query("python memory")
    assert q2 == "enhanced Q2"
    mock_ollama.assert_called_once()


@patch("src.memory.call_ollama")
def test_update_interaction_filters_no_insights(mock_ollama, memory_manager, mock_rag_manager):
    # Mocking call_ollama to return sequentially the intent, then the insights
    # Intent extraction returns 'intent: learn'
    # Insight extraction returns 'No insights' (case insensitive testing)
    mock_ollama.side_effect = ["intent: learn", "No Insights."]
    
    memory_manager.update_after_interaction("hello", "hi")
    
    assert memory_manager.extended_stm == "intent: learn"
    # Ensure write_insight was NEVER called because of 'no insights' filtering
    mock_rag_manager.write_insight.assert_not_called()
    mock_rag_manager.add_to_history.assert_called_with("hello", "hi")


@patch("src.memory.call_ollama")
def test_update_interaction_writes_valid_insight(mock_ollama, memory_manager, mock_rag_manager):
    # Intent extraction -> 'intent: buy'
    # Insight extraction -> '- user is broke'
    mock_ollama.side_effect = ["intent: buy", "- user is broke"]
    
    memory_manager.update_after_interaction("test", "test")
    
    mock_rag_manager.write_insight.assert_called_once_with("user is broke")
