"""
Módulo del Agente (LLM - Large Language Model).

Propósito: Conectar con un modelo de lenguaje local (Ollama) para generar 
respuestas conversacionales basadas en entrada del usuario.

Características:
- Comunicación con Ollama local vía HTTP REST API
- Tool Use / Function Calling: El LLM puede invocar herramientas externas
- Manejo de respuestas normales y respuestas con tool calling

Flujo (sin tools):
1. Recibe texto transcrito (user_text) del nodo STT
2. Construye prompt con system_prompt + user_text
3. Envía a Ollama (local) vía HTTP REST API
4. Ollama genera respuesta (puede incluir <think> tags)
5. Retorna respuesta al nodo TTS para reproducción

Flujo (con tools):
1. Recibe user_text del nodo STT
2. LLM analiza entrada y decide si necesita una herramienta
3. Si SI → LLM retorna JSON con tool_name + tool_input
4. Agent ejecuta la herramienta
5. LLM recibe resultado y genera respuesta final
6. Retorna respuesta al nodo TTS

Nota: Filtrado de tags de <think> ocurre en src/graph.py en el nodo TTS.
"""

import requests
import json
from config import OLLAMA_URL, OLLAMA_MODEL
from .tools import get_tools_description


SYSTEM_PROMPT = (
    "Eres un asistente de voz breve y útil que puede usar herramientas externas.\n"
    "Responde en español, de forma concisa, sin emojis ni asteriscos.\n\n"
    "INSTRUCCIONES PARA TOOL USE:\n"
    "- Analiza la pregunta del usuario.\n"
    "- Si necesitas información: usa las herramientas disponibles.\n"
    "- Puedes usar MÚLTIPLES herramientas.\n"
    "- Si ya tienes suficiente información (o no necesitas herramientas), responde directamente.\n\n"
    "FORMATO CORRECTO DE ARGUMENTOS MÚLTIPLES:\n"
    "Cuando una herramienta requiere múltiples argumentos, DEBES usar este formato exacto:\n"
    '   {"type": "tool_call", "tool": "nombre_herramienta", "input": "\'arg1\':\'arg2\'"}\n'
    "Nota: Cada argumento va ENTRE COMILLAS SIMPLES y separados por ':' (dos puntos).\n"
    "Ejemplos:\n"
    '  - Un argumento: {"type": "tool_call", "tool": "calculator", "input": "\'2+2\'"}\n'
    '  - Dos argumentos: {"type": "tool_call", "tool": "read_file", "input": "\'config.py\':\'50\'"}\n'
    "FORMATO DE RESPUESTA:\n"
    "1. Si necesitas una herramienta, responde SOLO en JSON con el formato anterior.\n"
    "2. Si tienes toda la información, responde SOLO con texto (sin JSON).\n\n"
    f"{get_tools_description()}"
)


def call_ollama_decision(prompt: str) -> str:
    """
    LLAMADA A LLM: Genera respuesta a un prompt del usuario.
    
    Usado por:
    - Agent ReAct Loop (soporta histórico de tools)
    - Cada iteración del loop
    
    El LLM puede responder:
    a) Con texto normal → respuesta directa (sin herramientas)
    b) Con JSON tool_call → se ejecutará la herramienta y loop continuará
    
    Parámetros:
    - prompt: str - Puede ser user_text simple O user_text + histórico de tools
                    (el ReAct Loop se encarga de construir este prompt)
    
    Retorna:
    - str - Respuesta del LLM (puede ser texto normal o JSON con tool_call)
    
    Flujo:
    1. Enviar prompt a Ollama
    2. Ollama responde con texto normal O JSON tool_call
    3. Retornar respuesta (el caller decidirá qué hacer)
    
    Requisitos:
    - Ollama ejecutándose localmente
    - Modelo configurado en config.py (OLLAMA_MODEL)
    """
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nUsuario: {prompt}\nAsistente:"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.5},
        "think": True
    }
    
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        response_text = data.get("response", "").strip()
        return response_text
    
    except Exception as e:
        return f"Error comunicándose con Ollama: {e}"

