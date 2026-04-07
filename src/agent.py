import requests
import re
import json
import time
from .utils import Config
config = Config()

API_URL = config.get("API_URL")
RESPONSE_MODEL = config.get("RESPONSE_MODEL")
AGENT_EXTRA_PROMPT = config.get("AGENT_EXTRA_PROMPT")
USER_NAME = config.get("USER_NAME")
verbose_mode = config.get("verbose_mode")

from .Tools.registry import get_tools_description

_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)


# 1. PROMPT FOR TOOL DECISOR NODE
TOOL_DECISOR_SYSTEM_PROMPT = (
    "You are the central brain of an assistant. Your sole function is to decide what type of action "
    "should handle the user's request, based on the conversation history and the information found (RAG).\n\n"
    "{formatted_tools_context}"
    "AVAILABLE ACTIONS:\n"
    "- 'tool': To use an external tool to get, create, or process information.\n"
    "- 'response': WHEN YOU ALREADY HAVE ENOUGH INFORMATION to answer the user (e.g., if the RAG Information is sufficient), or if the request is a simple chat.\n"
    "- 'exit': If the user explicitly requests to exit or close the program.\n"
    "CRITICAL RULE: If a tool has ALREADY been executed for the user's request (you can see the result in the TOOL CONTEXT), you MUST choose 'response' to finish the task. DO NOT choose 'tool' again if the task is already done.\n"
    "If the RAG information is sufficient to answer the user, choose 'response'.\n"
    "Respond ONLY with one of these words: tool, response, exit."
)


# 1.5 PROMPT FOR RAG DECISOR NODE
RAG_DECISOR_SYSTEM_PROMPT = (
    "You are an evaluator. Your only job is to determine if the provided RAG context is sufficient "
    "to answer the user's query. YOU MUST NOT PROVIDE EXPLANATIONS OR REASONS.\n\n"
    "RAG CONTEXT:\n"
    "```\n{rag_context}\n```\n\n"
    "If the context contains enough relevant information to fully or partially answer the user's request, respond 'yes'.\n"
    "If the context is irrelevant, empty, or does not help to answer the user, respond 'no'.\n"
    "CRITICAL RULE: Respond EXCLUSIVELY with the single word 'yes' or 'no'. Any other text is strictly forbidden."
)


# 2. PROMPT FOR TOOL GENERATION (Template)
TOOL_GENERATION_PROMPT_TEMPLATE = (
    "You are an expert agent in using external tools.\n"
    "Your goal is to generate the JSON to invoke the precise tool that resolves the user's request.\n\n"
    "HISTORY OF ACTIONS PERFORMED:\n"
    "```\n{history_context}\n```\n\n"
    "AVAILABLE TOOLS:\n"
    "{tools_desc}\n\n"
    "IMPORTANT INSTRUCTIONS FOR JSON OUTPUT:\n"
    "- You MUST respond with a single JSON object.\n"
    "- It MUST contain a \"tool\" key with the exact ID of the tool you want to use.\n"
    "- Any required arguments defined in the tool's Schema MUST be included directly alongside the \"tool\" key.\n"
    "- Example format:\n"
    "  {{\n"
    "    \"tool\": \"server_name.tool_name\",\n"
    "    \"parameter_name\": \"parameter_value\"\n"
    "  }}\n"
    "- Respond ONLY with the JSON. No explanatory text.\n"
)


# 3. PROMPT FOR FINAL RESPONSE
FINAL_RESPONSE_SYSTEM_PROMPT = (
    "Your task is to generate the final response for the user.\n"
    f"The user's name is: {USER_NAME}\n"
    "Requested Language: {language}.\n\n"
    "PERSONALITY:\n"
    f"{AGENT_EXTRA_PROMPT.strip() if AGENT_EXTRA_PROMPT else 'Be helpful and direct.'}\n\n"
    "RULES:\n"
    "- Respond in a short and concise manner.\n"
    "- DO NOT mention 'I have used tool X', 'the RAG says' or 'the context says', simply provide the information as your own knowledge.\n"
    "- DO NOT roleplay.\n"
    "- DO NOT use md format.\n"
    "- DO NOT use embedded links.\n"
    "- DO NOT respond with HTTP links.\n"
    "- DO NOT highlight keywords or phrases in backticks.\n"
    "- IF THERE IS RAG CONTEXT OR TOOL CONTEXT: The definitive answer and absolute truth is found there. Use it to respond.\n"
    "- IF THERE IS HISTORY CONTEXT: It is only additional context from previous interactions.\n\n"
    "- ONLY RESPOND IN NATURAL LANGUAGE.\n"
    "{formatted_rag_context}"
    "{formatted_tools_context}"
    "{formatted_history_context}"
)


# 4. PROMPT FOR ACKNOWLEDGEMENT
ACKNOWLEDGEMENT_SYSTEM_PROMPT = (
    "You are a voice assistant. You have just been activated by your wake word.\n"
    "Generate an extremely short acknowledgement phrase (1 or 2 words) in singular to indicate that you are listening.\n"
    "Requested Language: {language}.\n"
    "PERSONALITY:\n"
    f"{AGENT_EXTRA_PROMPT.strip() if AGENT_EXTRA_PROMPT else 'Kind and helpful.'}\n\n"
    "EXAMPLES:\n"
    "- 'Yes?'\n"
    "- 'Tell me'\n"
    "- 'I am listening'\n"
    "- 'What do you need?'\n"
    "- 'Dime'\n"
    "- 'Te escucho'\n\n"
    "Respond ONLY with the phrase, without anything else."
)


# 4.5 PROMPT FOR WAITING
WAITING_SYSTEM_PROMPT = (
    "You are a voice assistant. The user has requested a task that requires some processing time.\n"
    "Generate an extremely short waiting phrase (maximum 4 words) to indicate that you are on it.\n"
    "Requested Language: {language}.\n"
    "PERSONALITY:\n"
    f"{AGENT_EXTRA_PROMPT.strip() if AGENT_EXTRA_PROMPT else 'Kind and helpful.'}\n\n"
    "EXAMPLES:\n"
    "- 'Just a moment'\n"
    "- 'Working on it'\n"
    "- 'Let me check'\n"
    "- 'I am on it'\n"
    "Respond ONLY with the phrase, without anything else."
)


# 5. PROMPT FOR FAREWELL
FAREWELL_SYSTEM_PROMPT = (
    "You are a voice assistant. The user has requested to close the program or is saying goodbye.\n"
    "Generate a very short farewell phrase (maximum 3 words).\n"
    "Requested Language: {language}.\n"
    "PERSONALITY:\n"
    f"{AGENT_EXTRA_PROMPT.strip() if AGENT_EXTRA_PROMPT else 'Kind and helpful.'}\n\n"
    "EXAMPLES:\n"
    "- 'See you later!'\n"
    "- 'See you soon'\n"
    "- 'Disconnecting, bye!'\n"
    "- 'Hasta luego'\n"
    "- 'Adiós'\n"
    "- 'Hasta la proxima!'\n"
    "- 'Nos veremos!\n"
    "Respond ONLY with the phrase, without anything else."
)


# 6. PROMPT FOR WAKEWORD CONFIG
WAKEWORD_CONFIG_SYSTEM_PROMPT = (
    "You are a voice assistant guiding the user to configure your wake word.\n"
    "Generate a very short phrase EXACTLY according to the user's instruction.\n"
    "CRITICAL RULE: NEVER GREET THE USER. NEVER say 'Hello', 'Hi', 'Hola', or any greeting whatsoever. Get straight to the instruction.\n"
    "Requested Language: {language}.\n"
    "PERSONALITY:\n"
    f"{AGENT_EXTRA_PROMPT.strip() if AGENT_EXTRA_PROMPT else 'Kind and helpful.'}\n\n"
    "WHAT YOU HAVE ALREADY SAID PREVIOUSLY (Do not repeat the same exact phrases!):\n"
    "{history}\n\n"
    "Respond ONLY with the phrase to speak, without quotes, emojis, or descriptions of actions."
)



# 7. PROMPT FOR LTM INSIGHT EXTRACTION
INSIGHT_EXTRACTION_PROMPT = (
    "Extract persistent traits, preferences, or goals from the user's query.\n"
    "Only extract long-term insights (e.g., 'beginner in programming', 'interested in AI', 'speaks spanish').\n"
    "DO NOT extract one-time questions, facts, or redundant data.\n"
    "If there are no meaningful long-term insights in the query, respond EXACTLY with: no insights\n"
    "Format as a concise bulleted list of keywords/short phrases.\n\n"
    "USER QUERY:\n{query}\n\n"
    "Respond ONLY with the insights or 'no insights', without additional text."
)


# 8. PROMPT FOR STM EXTENDED INTENT EXTRACTION
INTENT_EXTRACTION_PROMPT = (
    "Extract the core intent and constraints from the user query.\n"
    "Format exactly as:\n"
    "- intent: [core intent]\n"
    "- constraint: [key constraints if any]\n\n"
    "USER QUERY:\n{query}\n\n"
    "Respond ONLY with the formatted intent/constraint."
)


# 9. PROMPT FOR DUAL QUERY ENRICHMENT (Q2)
QUERY_ENRICHMENT_PROMPT = (
    "You are a query enhancer. Enhance the user's original query (Q1) using the short-term context (STM).\n"
    "Combine the original query with the STM intent to create a more specific, independent search query (Q2).\n"
    "CRITICAL RULE: DO NOT answer the query. Just output the enhanced query string.\n\n"
    "ORIGINAL QUERY (Q1): {query}\n"
    "SHORT-TERM CONTEXT (STM):\n{stm_context}\n\n"
    "ENHANCED QUERY (Q2):"
)


def get_tool_decisor_prompt(tools_context: str = "") -> str:
    formatted_tools = f"TOOL CONTEXT:\n```\n{tools_context}\n```\n\n" if tools_context else ""
    return TOOL_DECISOR_SYSTEM_PROMPT.format(
        formatted_tools_context=formatted_tools
    )


def get_rag_decisor_prompt(rag_context: str) -> str:
    return RAG_DECISOR_SYSTEM_PROMPT.format(rag_context=rag_context)


def get_tool_agent_prompt(tools_desc: str, rag_context: str = "", history_context: str = "") -> str:
    return TOOL_GENERATION_PROMPT_TEMPLATE.format(tools_desc=tools_desc, history_context=history_context)


def get_final_response_prompt(tools_context: str = "", history_context: str = "", rag_context: str = "", language: str = "English") -> str:
    formatted_rag = f"RAG CONTEXT (Documents):\n```\n{rag_context}\n```\n\n" if rag_context else ""
    formatted_tools = f"TOOL CONTEXT (Definitive Information):\n```\n{tools_context}\n```\n\n" if tools_context else ""
    formatted_history = f"HISTORY CONTEXT (Additional Context):\n```\n{history_context}\n```" if history_context else ""
    return FINAL_RESPONSE_SYSTEM_PROMPT.format(
        formatted_history_context=formatted_history, 
        formatted_rag_context=formatted_rag,
        formatted_tools_context=formatted_tools,
        language=language
    )


def get_acknowledgement_prompt(language: str = "English") -> str:
    return ACKNOWLEDGEMENT_SYSTEM_PROMPT.format(language=language)


def get_waiting_prompt(language: str = "English") -> str:
    return WAITING_SYSTEM_PROMPT.format(language=language)


def get_farewell_prompt(language: str = "English") -> str:
    return FAREWELL_SYSTEM_PROMPT.format(language=language)


def get_wakeword_prompt(language: str = "English", history: str = "") -> str:
    return WAKEWORD_CONFIG_SYSTEM_PROMPT.format(language=language, history=history)


def get_insight_extraction_prompt(query: str) -> str:
    return INSIGHT_EXTRACTION_PROMPT.format(query=query)


def get_intent_extraction_prompt(query: str) -> str:
    return INTENT_EXTRACTION_PROMPT.format(query=query)


def get_query_enrichment_prompt(query: str, stm_context: str) -> str:
    return QUERY_ENRICHMENT_PROMPT.format(query=query, stm_context=stm_context)


def clean_emojis(text: str) -> str:
    """Removes emojis from the text."""
    cleaned_text = re.sub(r'[*\$]', '', text)
    return re.sub(r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff]', '', cleaned_text)


def clean_think_tags(text: str) -> str:
    """Removes <think> tags."""
    if not text: return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think\s*/?>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def call_ollama(prompt: str, model: str = RESPONSE_MODEL, system_prompt: str = None, json_mode: bool = False, temperature: float = 0.3, think: bool = False) -> str:
    """
    Calls the Ollama API and logs the generation time.
    """
    full_prompt = f"{system_prompt}\n\nUser/Context: {prompt}\nAssistant:" if system_prompt else prompt
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": temperature},
        "think": think
    }
    
    if json_mode:
        payload["format"] = "json"

    start_time = time.time()
    try:
        r = requests.post(API_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        response = data.get("response", "").strip()
        
        duration = time.time() - start_time
        print(f"[🤖 Ollama] Generation completed in {duration:.2f}s (Model: {model})")
        
        return response
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"[🤖 Ollama] ❌ Error after {duration:.2f}s: {e}")
        return f"Ollama API Error: {e}"


def call_ollama_stream(prompt: str, model: str = RESPONSE_MODEL, system_prompt: str = None, json_mode: bool = False, temperature: float = 0.3, think: bool = False):
    """
    Makes a call to the Ollama API requesting a stream and yields it.
    """
    full_prompt = f"{system_prompt}\n\nUser/Context: {prompt}\nAssistant:" if system_prompt else prompt
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": True,
        "options": {"temperature": temperature},
        "think": think
    }
    
    if json_mode:
        payload["format"] = "json"

    try:
        r = requests.post(API_URL, json=payload, timeout=120, stream=True)
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                yield data.get("response", "")
    except Exception as e:
        print(f"[🤖 Ollama Stream] ❌ Error: {e}")

if __name__ == "__main__":
    # Ollama stream test
    test_prompt = "Hello, explain the transformers archiquectury"
    print(f"--- Testing call_ollama_stream with: '{test_prompt}' ---")
    try:
        for chunk in call_ollama_stream(test_prompt):
            print(chunk, end="", flush=True)
        print("\n--- End of test ---")
    except Exception as e:
        print(f"\nError in test: {e}")


