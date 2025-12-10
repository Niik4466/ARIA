"""
Módulo del Agente (LLM - Large Language Model).

Propósito: Conectar con un modelo de lenguaje local (Ollama) para generar 
respuestas conversacionales y decisiones de flujo.

Incluye:
- Prompts para Decisor (Router).
- Prompts para Agentes de Herramientas (Generadores JSON).
- Prompt para Nodo de Respuesta Final.
- Llamada a API de Ollama.
"""

import requests
import re
from config import API_URL, RESPONSE_MODEL, AGENT_EXTRA_PROMPT
from .Tools.registry import get_tools_description

# 1. PROMPT PARA EL NODO DECISOR DE HERRAMIENTAS
TOOL_DECISOR_SYSTEM_PROMPT = (
    "Eres el cerebro central de un asistente. Tu única función es decidir qué tipo de experto "
    "debe manejar la petición del usuario, basándote en el historial de la conversación y la información encontrada (RAG).\n\n"
    "INFORMACIÓN ENCONTRADA:\n"
    "```\n{rag_context}\n```\n\n"
    "HISTORIAL RECENTE:\n"
    "```\n{history_context}\n```\n\n"
    "CATEGORÍAS DISPONIBLES:\n"
    "- 'search': Para buscar información actual en internet, noticias, clima.\n"
    "- 'os': Para interactuar con el sistema operativo (crear archivos o programas, ejecutar comandos, media).\n"
    "- 'basic': Para cálculos matemáticos o consulta de hora.\n"
    "- 'response': CUANDO YA TENGAS SUFICIENTE INFORMACIÓN para responder al usuario (por ejemplo, si la Información RAG es suficiente), o si la petición es un saludo/charla simple.\n\n"
    "REGLA: Si falta información crítica, elige una herramienta. Si ya se ejecutaron herramientas "
    "y tienes el resultado en el historial, debes elegir 'response'.\n"
    "Si la información del RAG es suficiente para responder al usuario, elige 'response'.\n"
    "Si el usuario pide crear, leer, escribir, ejecutar o eliminar un archivo o programa, elige 'os'.\n"
    "Responde SOLO con una de estas palabras: search, os, basic, response."
)

# 1.5 PROMPT PARA EL NODO DECISOR DE RAG
RAG_DECISOR_SYSTEM_PROMPT = (
    "Tu función es determinar si la consulta del usuario requiere información de documentos internos almacenados en el RAG.\n\n"
    "CATEGORÍAS RAG DISPONIBLES:\n"
    "```{rag_categories_desc}```\n\n"
    "Si la consulta se relaciona CLARAMENTE con alguna de estas categorías, responde con el NOMBRE de la categoría.\n"
    "Si no se relaciona o es una charla general, responde 'none'.\n"
    "Responde SOLO con una palabra: el nombre de la categoría o 'none'."
)


# 2. PROMPT PARA GENERACIÓN DE TOOLS (Template)
TOOL_GENERATION_PROMPT_TEMPLATE = (
    "Eres un agente experto en el uso de herramientas de tipo: {category}.\n"
    "Tu objetivo es generar el JSON para invocar la herramienta precisa que resuelva la petición.\n\n"
    "CONTEXTO RAG (Información recuperada):\n"
    "```\n{rag_context}\n```\n\n"
    "HISTORIAL DE ACCIONES REALIZADAS:\n"
    "```\n{history_context}\n```\n\n"
    "HERRAMIENTAS DISPONIBLES:\n"
    "{tools_desc}\n\n"
    "FORMATO DE SALIDA (Estricto JSON):\n"
    "- {{\"tool\": \"nombre\", \"arg1\": \"valor\"}}\n"
    "- Responde SOLO con el JSON. Sin texto explicativo.\n"
)

# 3. PROMPT PARA RESPUESTA FINAL
FINAL_RESPONSE_SYSTEM_PROMPT = (
    "Eres un asistente de voz amable y eficiente. Tu tarea es generar la respuesta final para el usuario "
    "basándote en la información recopilada por las herramientas y documentos.\n\n"
    "PERSONALIDAD:\n"
    f"{AGENT_EXTRA_PROMPT.strip() if AGENT_EXTRA_PROMPT else 'Sé útil y directo.'}\n\n"
    "REGLAS:\n"
    "- Responde de forma corta y concisa.\n"
    "- Usa la información del Historial de Herramientas y el Contexto RAG para responder.\n"
    "- Habla en español natural.\n"
    "- NO menciones 'he usado la herramienta X', 'el RAG dice' o 'el contexto dice', simplemente da la información como conocimiento propio.\n"
    "- NO hagas roleplay\n"
    "- NO uses formato md\n"
    "- NO utilices enlaces enbebidos\n"
    "- NO respondas enlaces HTTP\n"
    "- NO remarques palabras clave o frases entre comillas invertidas\n"
    "- SI HAY CONTEXTO RAG: Úsalo para responder, la información entre comillas invertidas es la verdad absoluta.\n\n"
    "- SOLO RESPONDER EN LENGUAJE NATURAL\n"
    "CONTEXTO RAG (Documentos):\n"
    "```\n{rag_context}\n```\n\n"
    "Historial de Herramientas e Información:\n"
    "```\n{history_context}\n```"
)


def get_tool_decisor_prompt(rag_context: str = "", history_context: str = "") -> str:
    return TOOL_DECISOR_SYSTEM_PROMPT.format(rag_context=rag_context, history_context=history_context)

def get_rag_decisor_prompt(rag_categories_desc: str) -> str:
    return RAG_DECISOR_SYSTEM_PROMPT.format(rag_categories_desc=rag_categories_desc)

def get_tool_agent_prompt(category: str, rag_context: str = "", history_context: str = "") -> str:
    tools_desc = get_tools_description(category)
    return TOOL_GENERATION_PROMPT_TEMPLATE.format(category=category, tools_desc=tools_desc, rag_context=rag_context, history_context=history_context)

def get_final_response_prompt(history_context: str, rag_context: str = "") -> str:
    return FINAL_RESPONSE_SYSTEM_PROMPT.format(history_context=history_context, rag_context=rag_context)

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

def call_ollama(prompt: str, model: str = RESPONSE_MODEL, system_prompt: str = None, json_mode: bool = False, temperature: float = 0.3) -> str:
    """
    Llamada a Ollama.
    """
    full_prompt = f"{system_prompt}\n\nUsuario/Contexto: {prompt}\nAsistente:" if system_prompt else prompt
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": temperature},
        "think": True 
    }
    
    if json_mode:
        payload["format"] = "json"

    try:
        r = requests.post(API_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    
    except Exception as e:
        return f"Error API Ollama: {e}"


