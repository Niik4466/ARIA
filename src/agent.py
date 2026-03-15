import requests
import re
import json
import time
from config import API_URL, RESPONSE_MODEL, AGENT_EXTRA_PROMPT, USER_NAME
from .Tools.registry import get_tools_description

from config import verbose_mode
_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)


# 1. PROMPT PARA EL NODO DECISOR DE HERRAMIENTAS
TOOL_DECISOR_SYSTEM_PROMPT = (
    "You are the central brain of an assistant. Your sole function is to decide what type of expert "
    "should handle the user's request, based on the conversation history and the information found (RAG).\n\n"
    "{formatted_rag_context}"
    "{formatted_tools_context}"
    "AVAILABLE CATEGORIES:\n"
    "- 'search': To look for current information on the internet, news, weather.\n"
    "- 'os': To interact with the operating system (create files or programs, execute commands, utilize media buttons).\n"
    "- 'basic': For mathematical calculations or querying the time.\n"
    "- 'autoconfig': To configure the assistant, mainly with voice or wakeword changes.\n"
    "- 'response': WHEN YOU ALREADY HAVE ENOUGH INFORMATION to answer the user (e.g., if the RAG Information is sufficient), or if the request is a greeting/simple chat.\n"
    "- 'exit': If the user requests to exit or close the program.\n"
    "CRITICAL RULE: If a tool has ALREADY been executed for the user's request (you can see the result in the TOOL CONTEXT), you MUST choose 'response' to finish the task. DO NOT choose a tool category again if the task is already done.\n"
    "If the RAG information is sufficient to answer the user, choose 'response'.\n"
    "If the user asks to create, read, write, execute or delete a file or program, choose 'os'.\n"
    "Respond ONLY with one of these words: search, os, basic, autoconfig, response."
)


# 1.5 PROMPT PARA EL NODO DECISOR DE RAG
RAG_DECISOR_SYSTEM_PROMPT = (
    "Your function is to determine if the user's query requires information from internal documents stored in the RAG.\n\n"
    "AVAILABLE RAG CATEGORIES:\n"
    "```{rag_categories_desc}```\n\n"
    "If the query CLEARLY relates to any of these categories, respond with the category NAME.\n"
    "If it does not relate or is general chat, respond 'none'.\n"
    "Respond ONLY with one word: the category name or 'none'."
)


# 2. PROMPT PARA GENERACIÓN DE TOOLS (Template)
TOOL_GENERATION_PROMPT_TEMPLATE = (
    "You are an expert agent in using tools of type: {category}.\n"
    "Your goal is to generate the JSON to invoke the precise tool that resolves the request.\n\n"
    "RAG CONTEXT (Retrieved information):\n"
    "```\n{rag_context}\n```\n\n"
    "HISTORY OF ACTIONS PERFORMED:\n"
    "```\n{history_context}\n```\n\n"
    "AVAILABLE TOOLS:\n"
    "{tools_desc}\n\n"
    "OUTPUT FORMAT (Strict JSON):\n"
    "- {{\"tool\": \"name\", \"arg1\": \"value\"}}\n"
    "- {{\"tool\": \"name\"}}\n"
    "- Respond ONLY with the JSON. No explanatory text.\n"
)


# 3. PROMPT PARA RESPUESTA FINAL
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


# 4. PROMPT PARA ACKNOWLEDGEMENT (ACUSE DE RECIBO)
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


# 4.5 PROMPT PARA WAITING (ESPERA)
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


# 5. PROMPT PARA FAREWELL (DESPEDIDA)
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


# 6. PROMPT PARA WAKEWORD CONFIG
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



def get_tool_decisor_prompt(tools_context: str = "", rag_context: str = "", history_context: str = "") -> str:
    formatted_rag = f"FOUND INFORMATION:\n```\n{rag_context}\n```\n\n" if rag_context else ""
    formatted_tools = f"TOOL CONTEXT:\n```\n{tools_context}\n```\n\n" if tools_context else ""
    return TOOL_DECISOR_SYSTEM_PROMPT.format(
        formatted_rag_context=formatted_rag, 
        formatted_tools_context=formatted_tools
    )


def get_rag_decisor_prompt(rag_categories_desc: str) -> str:
    return RAG_DECISOR_SYSTEM_PROMPT.format(rag_categories_desc=rag_categories_desc)


def get_tool_agent_prompt(category: str, rag_context: str = "", history_context: str = "") -> str:
    tools_desc = get_tools_description(category)
    return TOOL_GENERATION_PROMPT_TEMPLATE.format(category=category, tools_desc=tools_desc, rag_context=rag_context, history_context=history_context)


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


def clean_emojis(text: str) -> str:
    """Elimina emojis del texto."""
    cleaned_text = re.sub(r'[*\$]', '', text)
    return re.sub(r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff]', '', cleaned_text)


def clean_think_tags(text: str) -> str:
    """Elimina tags <think>."""
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
    Realiza una llamada a la API de Ollama pidiendo un flujo (stream) y lo retorna.
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
        print(f"\nError en la prueba: {e}")


