import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from unittest.mock import patch, MagicMock

def test_mcp_manager_init():
    from src.mcp.mcp_manager import MCPManager
    mock_rag = MagicMock()
    
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", unittest_mock_open=True):
            manager = MCPManager(rag_manager=mock_rag)
            assert manager is not None

def test_mcp_sync_rag():
    from src.mcp.mcp_manager import MCPManager
    mock_rag = MagicMock()
    
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", unittest_mock_open=True):
            manager = MCPManager(rag_manager=mock_rag)
        mock_client = MagicMock()
        mock_client.list_tools.return_value = [{"name": "tool_x", "description": "foo", "inputSchema": {}}]
        manager.clients = {"fake_server": mock_client} # Mock Client structure
        
        # get_tools wrapper will trigger sync and then query rag
        mock_rag.get_tools.return_value = [{"id": "fake_server.tool_x", "text": "foo"}]
        results = manager.get_tools("Find tool x")
        
        assert len(results) == 1
        assert results[0].id == "fake_server.tool_x"
