import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def test_clean_think_tags():
    from src.agent import clean_think_tags
    
    # Test valid removal
    text = "Here is my thought: <think>This should be removed</think> Result is this."
    assert clean_think_tags(text).strip() == "Here is my thought:  Result is this."
    
    # Test un-closed tag behavior
    text2 = "Normal text <think>Only opening"
    assert "<think>" not in clean_think_tags(text2)
    
    # Test multiple tags
    text3 = "<think>hide 1</think> show <think>hide 2</think>"
    assert clean_think_tags(text3).strip() == "show"

def test_clean_emojis():
    from src.agent import clean_emojis
    text = "Hello! 😊 Let's test 🚀"
    assert clean_emojis(text) == "Hello!  Let's test "

def test_prompts_generation():
    from src.agent import get_tool_agent_prompt, get_rag_decisor_prompt
    
    # Validate prompt templating runs without crashing
    tool_prompt = get_tool_agent_prompt("MyTools", "MyRag", "MyHistory")
    assert "MyTools" in tool_prompt
    
    rag_prompt = get_rag_decisor_prompt("ContextString")
    assert "ContextString" in rag_prompt
