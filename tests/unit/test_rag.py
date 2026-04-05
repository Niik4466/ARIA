import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from unittest.mock import patch, MagicMock

def test_rag_weighted_vote():
    from src.rag import RAGManager
    
    with patch("src.rag.chromadb.PersistentClient"), patch("src.rag.embedding_functions.SentenceTransformerEmbeddingFunction"):
        rag = RAGManager()
        
        # Test exact match logic
        docs = [
            {"category": "AI", "similarity": 0.8},
            {"category": "Hardware", "similarity": 0.5},
            {"category": "AI", "similarity": 0.3}
        ]
        
        best = rag.weighted_category_vote(docs)
        assert best == "AI" # 0.8 + 0.3 = 1.1 vs 0.5
        
        # Empty
        assert rag.weighted_category_vote([]) == "none"

def test_rag_query_history():
    from src.rag import RAGManager
    
    with patch("src.rag.chromadb.PersistentClient") as mock_chroma, \
         patch("src.rag.embedding_functions.SentenceTransformerEmbeddingFunction"):
        
        rag = RAGManager()
        
        # Mocking the internal collection
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = {"documents": [["Historical User context."]]}
        rag.history_collection = mock_col
        
        result = rag.query_history("Hello?")
        assert result == "Historical User context."
        mock_col.query.assert_called()

def test_rag_query_category():
    from src.rag import RAGManager
    with patch("src.rag.chromadb.PersistentClient"), \
         patch("src.rag.embedding_functions.SentenceTransformerEmbeddingFunction"):
        
        rag = RAGManager()
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["Document content"]],
            "metadatas": [[{"source": "test.md"}]]
        }
        rag.docs_collection = mock_col
        
        result = rag.query_category("docs", "Question")
        assert "Excerpt from test.md" in result
        assert "Document content" in result
