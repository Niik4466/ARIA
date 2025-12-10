"""
Módulo del Grafo de Procesamiento (LangGraph).

Propósito: Definir el pipeline de procesamiento de voz usando una arquitectura
orientada a decisiones y estados (Tool_Decisor -> Rag_Decisor -> Tool/Response).
"""

from typing import TypedDict, List, Literal
import re
import json
import numpy as np
from langgraph.graph import StateGraph, END

from config import DECISOR_MODEL, RESPONSE_MODEL
from .stt import FasterWhisperSTT
from .agent import (
    call_ollama, 
    clean_think_tags, 
    clean_emojis,
    get_tool_decisor_prompt,
    get_rag_decisor_prompt,
    get_tool_agent_prompt,
    get_final_response_prompt
)
from .tts import TTSEngine, run_tts_thread
from .Tools.registry import execute_tool
from .rag import rag_manager

# --- Inicialización del RAG ---
print("🔄 Inicializando sistema RAG...")
RAG_CATEGORIES = rag_manager.update()
RAG_CATEGORIES_DESC_STR = "\n".join([f"- '{k}': {v}" for k, v in RAG_CATEGORIES.items()])
if not RAG_CATEGORIES_DESC_STR:
    RAG_CATEGORIES_DESC_STR = "(No hay documentos disponibles)"
print("✅ RAG Inicializado.")


# --- Estado del Grafo ---
class GraphState(TypedDict, total=False):
    # Entrada
    frames: List[np.ndarray]
    
    # Estado acumulativo
    user_text: str
    history_context: str       # Texto acumulado con resultados de herramientas
    iteration_count: int       # Contador para seguridad (max loops)
    
    # RAG
    rag_category: str          # Categoría seleccionada del RAG
    rag_context: str           # Texto recuperado

    # Señales de control
    next_node: str             # 'tool_node' o 'response_node'
    selected_category: str     # 'search', 'os', 'basic', etc.
    
    # Salida final
    reply_text: str

# --- Instancias Globales ---
stt = FasterWhisperSTT()
tts = TTSEngine()


# --- Nodos ---

def stt_node(state: GraphState) -> GraphState:
    """
    NODO STT: Audio -> Texto
    Convierte audio a texto y inicializa el historial.
    """
    text = stt.transcribe(state["frames"]) or ""
    print(f"[🎤 STT] {text}")
    return {
        **state,
        "user_text": text,
        "history_context": "",
        "iteration_count": 0,
        "rag_context": "" # Reset de contexto previo
    }

def rag_decisor_node(state: GraphState) -> GraphState:
    """
    NODO DECISOR RAG: Define si se usa RAG y qué categoría.
    Se ejecuta ANTES del Tool Decisor para proveer contexto.
    """
    user_text = state["user_text"]
    
    # Si no hay categorías, saltamos
    if not RAG_CATEGORIES:
        return {**state, "rag_category": "none", "rag_context": ""}

    system_prompt = get_rag_decisor_prompt(RAG_CATEGORIES_DESC_STR)
    
    try:
        response = call_ollama(prompt=user_text, model=DECISOR_MODEL, system_prompt=system_prompt)
        # Limpieza de respuesta para obtener solo la categoría
        selected = response.strip().replace("'", "").replace('"', '').replace(".", "").lower()
    except Exception as e:
        print(f"Error en RAG Decisor: {e}")
        selected = "none"

    rag_context = ""
    # Verificar si la categoría existe en nuestros documentos
    # Hacemos matching laxo (lowercase)
    final_cat = "none"
    for cat in RAG_CATEGORIES.keys():
        if cat.lower() == selected:
            final_cat = cat
            break
            
    print(f"[📚 RAG Decisor] Selected: {final_cat}")

    if final_cat != "none":
        # Consultar RAG
        print(f"[📚 RAG] Buscando en '{final_cat}'...")
        rag_context = rag_manager.query_category(final_cat, user_text)
        if rag_context:
            print("[📚 RAG] Contexto recuperado.")
        else:
            print("[📚 RAG] No se encontró información relevante.")
            
    return {
        **state,
        "rag_category": final_cat,
        "rag_context": rag_context
    }

def tool_decisor_node(state: GraphState) -> GraphState:
    """
    NODO DECISOR DE HERRAMIENTAS: Texto + RAG -> Define Categoría de Agente
    """
    # Seguridad: Si excedemos iteraciones, forzamos respuesta
    if state.get("iteration_count", 0) >= 5:
        print("[⚠️ Decisor] Max iterations reached. Forcing response.")
        return {**state, "next_node": "response_node"}

    user_text = state["user_text"]
    rag_context = state.get("rag_context", "")
    
    if not user_text:
        return {**state, "agent_category": "general"}
    
    # Inyectamos contexto RAG en el prompt del decisor
    system_prompt = get_tool_decisor_prompt(rag_context=rag_context, history_context=state["history_context"])
    
    # Llamamos a Ollama
    try:
        response = call_ollama(prompt=user_text, model=DECISOR_MODEL, system_prompt=system_prompt)
        category = response.strip().lower()
        print(f"[🧠 Tool Decisor] said: {category}")
    except Exception as e:
        print(f"Error en Tool Decisor: {e}")
        category = "response"

    print(f"[🧠 Tool Decisor] said: {category}")   

    valid_categories = ['search', 'os', 'basic']
    # Buscamos si alguna categoría válida está en la respuesta
    found_category = next((c for c in valid_categories if c in category), "response")
    
    next_node = "tool_node" if found_category != "response" else "response_node"
    
    return {
        **state,
        "next_node": next_node,
        "selected_category": found_category
    }


def tool_node(state: GraphState) -> GraphState:
    """
    Genera el JSON de la herramienta y la ejecuta.
    """
    category = state["selected_category"]
    history = state["history_context"]
    user_text = state["user_text"]
    rag_context = state.get("rag_context", "")

    # 1. Feedback verbal
    feedbacks = {
        "search": "Dame un momento, lo busco.",
        "os": "Voy a hacer eso.",
        "basic": "Calculando..."
    }
    run_tts_thread(feedbacks.get(category, "Un momento."), tts)
    
    # 2. Generar JSON
    system_prompt = get_tool_agent_prompt(category=category, rag_context=rag_context, history_context=history)
    
    print(f"[🔧 ToolGen] Generando JSON para {category}...")
    response_json_str = call_ollama(
        prompt=user_text,
        model=RESPONSE_MODEL,
        system_prompt=system_prompt,
        json_mode=True 
    )
    
    tool_result_str = ""
    tool_name = "unknown"
    
    try:
        json_match = re.search(r'\{.*\}', response_json_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            tool_name = data.get("tool")
            tool_args = {k: v for k, v in data.items() if k != "tool"}
            
            print(f"[🔨 Ejecutando] {tool_name} {tool_args}")
            res = execute_tool(tool_name, tool_args)
            
            tool_result_str = f"Tool '{tool_name}' ejecutada. Resultado: {res}"
        else:
            tool_result_str = f"Error: No se generó JSON válido. Respuesta: {response_json_str}"
            
    except Exception as e:
        tool_result_str = f"Error ejecutando herramienta: {e}"
    print(f"response_json_str: {response_json_str}")
    print(f"[✅ Resultado] {tool_result_str}")
    
    new_history = history + f"\n[Acción] {tool_name} -> {tool_result_str}"
    
    return {
        **state,
        "history_context": new_history,
        "iteration_count": state["iteration_count"] + 1
    }


def response_node(state: GraphState) -> GraphState:
    """
    Genera la respuesta final para el usuario.
    """
    history = state["history_context"]
    rag_context = state.get("rag_context", "")
    user_text = state["user_text"]
    
    print(f"[✨ Response] Generando respuesta final...")

    response = call_ollama(
        prompt=user_text, 
        model=RESPONSE_MODEL,
        system_prompt=get_final_response_prompt(rag_context=rag_context, history_context=history)
    )
    
    return {**state, "reply_text": response}


def tts_node(state: GraphState) -> GraphState:
    """Reproduce la respuesta final."""
    text = state.get("reply_text", "")
    clean_text = clean_emojis(clean_think_tags(text))
    
    if clean_text:
        print(f"[🔊 TTS] {clean_text[:50]}...")
        tts.speak(clean_text)
        
    return state


# --- Grafo ---

def route_decision(state: GraphState) -> Literal["tool_node", "response_node"]:
    """Función para el Conditional Edge."""
    return state["next_node"]

def build_graph():
    g = StateGraph(GraphState)
    
    g.add_node("stt", stt_node)
    g.add_node("rag_decisor", rag_decisor_node)
    g.add_node("tool_decisor", tool_decisor_node)
    g.add_node("tool_node", tool_node)
    g.add_node("response_node", response_node)
    g.add_node("tts", tts_node)
    
    g.set_entry_point("stt")
    
    # Flujo: STT -> RAG Decisor -> Tool Decisor
    g.add_edge("stt", "rag_decisor")
    g.add_edge("rag_decisor", "tool_decisor")
    
    # Decision -> Tool o Response
    g.add_conditional_edges(
        "tool_decisor",
        route_decision
    )
    
    # Tool -> Vuelve a Tool Decisor (Loop)
    g.add_edge("tool_node", "tool_decisor")
    
    # Response -> TTS -> End
    g.add_edge("response_node", "tts")
    g.add_edge("tts", END)
    
    return g.compile()

