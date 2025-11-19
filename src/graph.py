"""
Módulo del Grafo de Procesamiento (LangGraph).

Propósito: Definir el pipeline de procesamiento de voz de extremo a extremo
con soporte para Tool Use (Function Calling) mediante ReAct Loop.

Arquitectura ReAct Loop (Reasoning + Acting):
┌─────────────────────────────────────────────────────────────┐
│                        STT Node                             │
│                   frames → user_text                        │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  Agent ReAct Node (LOOP)                    │
│                                                             │
│  1. Recibe: user_text + histórico de tools ejecutadas      │
│  2. LLM decide: ¿Necesita otra herramienta?               │
│     - SÍ  → Ejecuta tool internamente → Vuelve al paso 1  │
│     - NO  → Genera respuesta final → Sale del loop        │
│  3. Límite: máximo 5 iteraciones (evita loops infinitos)  │
│  4. Salida: reply_text (respuesta final limpia)           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   TTS Node                                  │
│              reply_text → audio reproducido                │
└─────────────────────────────────────────────────────────────┘

Ventajas:
✅ Múltiples tools por pregunta
✅ Razonamiento iterativo
✅ Context awareness entre iteraciones
✅ Límite de seguridad (máx 5 tools)
✅ Fácil de debuggear y extender
"""

from typing import TypedDict, List
import re
import json
import numpy as np
from langgraph.graph import StateGraph, END

from .stt import FasterWhisperSTT
from .agent import call_ollama_decision
from .tts import TTSEngine
from .tools import execute_tool


class GraphState(TypedDict, total=False):
    """
    Estado del Grafo con ReAct Loop.
    
    Campos:
    - frames: List[np.ndarray]
        Entrada: frames de audio PCM int16 del micrófono
    
    - user_text: str
        Salida STT: texto transcrito por Whisper
    
    - tool_calls_history: List[dict]
        Histórico de herramientas ejecutadas: [{"tool": "name", "input": "x", "result": "y"}, ...]
        Se construye iterativamente en el loop
    
    - tool_name: str
        Intermedio: nombre de la herramienta actual (en iteración actual)
    
    - tool_input: str
        Intermedio: input para la herramienta actual
    
    - tool_result: str
        Intermedio: resultado de ejecutar la herramienta actual
    
    - iteration_count: int
        Contador de iteraciones en el loop (máx 5)
    
    - reply_text: str
        Salida final: respuesta del LLM (limpia de <think> tags)
    """
    # Entrada
    frames: List[np.ndarray]
    # STT output
    user_text: str
    # ReAct Loop state
    tool_calls_history: List[dict]
    tool_name: str
    tool_input: str
    tool_result: str
    iteration_count: int
    # Salida final
    reply_text: str


# Instancias globales (se crean una sola vez)
stt = FasterWhisperSTT()      # Modelo Whisper para STT
tts = TTSEngine()              # Engine para TTS con edge-tts


def stt_node(state: GraphState) -> GraphState:
    """
    Nodo STT (Speech-To-Text): Convierte frames de audio a texto.
    
    Entrada:
    - state["frames"]: List[np.ndarray] - Frames de audio PCM int16
    
    Proceso:
    1. Pasar frames a FasterWhisperSTT.transcribe()
    2. Retorna lista de arrays que se concatenan en audio
    3. Se genera WAV en memoria (sin disco)
    4. Whisper transcribe a texto en español
    
    Salida:
    - state["user_text"]: str - Texto transcrito
    - Resto del estado se preserva
    
    Retorna: Nuevo estado con user_text añadido
    """
    text = stt.transcribe(state["frames"])
    return {**state, "user_text": text}


def agent_react_node(state: GraphState) -> GraphState:
    """
    NODO AGENT REACT: Implementa el patrón ReAct Loop (Reasoning + Acting).
    
    Este nodo es el corazón del sistema. Implementa un loop iterativo donde:
    1. El agente razona sobre la pregunta y el histórico de tools
    2. Decide: ¿Necesito otra herramienta? ¿Ya puedo responder?
    3. Si necesita tool: ejecuta, añade al histórico, vuelve al paso 1
    4. Si puede responder: retorna respuesta final
    
    Entrada:
    - state["user_text"]: str - Pregunta del usuario
    - state["tool_calls_history"]: List[dict] - Histórico de tools ejecutadas
    - state["iteration_count"]: int - Número de iteraciones realizadas
    
    Proceso (Loop):
    ```
    Para cada iteración (máx 5):
        1. Construir prompt con: user_text + histórico de tools
        2. LLM analiza y responde:
           a) {"type": "tool_call", "tool": X, "input": Y}
              → Ejecutar tool, añadir resultado al histórico, continuar loop
           b) Texto normal
              → Respuesta final, salir del loop
        3. Limitar a máx 5 iteraciones (evita loops infinitos)
    ```
    
    Salida:
    - state["reply_text"]: str - Respuesta final del LLM
    - state["tool_calls_history"]: List[dict] - Histórico completo de ejecuciones
    - state["iteration_count"]: int - Iteraciones totales usadas
    
    Ejemplo de ejecución:
    Usuario: "¿Cuál es la hora en Nueva York y qué temperatura hace?"
    
    Iteración 1:
      LLM: {"type": "tool_call", "tool": "execute_command", "input": "Get-Date"}
      Tool: "14:30:45"
      History: [{"tool": "execute_command", "input": "Get-Date", "result": "14:30:45"}]
    
    Iteración 2:
      LLM recibe: pregunta + histórico de iteración 1
      LLM: {"type": "tool_call", "tool": "weather", "input": "Nueva York"}
      Tool: "Nublado, 15°C"
      History: [..., {"tool": "weather", "input": "Nueva York", "result": "Nublado, 15°C"}]
    
    Iteración 3:
      LLM recibe: pregunta + histórico completo
      LLM: "En Nueva York son las 14:30 y hace 15°C con cielo nublado"
      Exit loop, return reply_text
    
    Retorna: Nuevo estado con reply_text y tool_calls_history actualizado
    """
    MAX_ITERATIONS = 5
    iteration_count = state.get("iteration_count", 0)
    tool_calls_history = state.get("tool_calls_history", [])
    user_text = state["user_text"]
    
    print(f"\n🔄 ReAct Loop - Iteración {iteration_count + 1}/{MAX_ITERATIONS}")
    
    while iteration_count < MAX_ITERATIONS:
        iteration_count += 1
        
        # Construir prompt con histórico de tools ejecutadas
        history_text = ""
        if tool_calls_history:
            history_text = "\n\nHistórico de herramientas ejecutadas:\n"
            for i, call in enumerate(tool_calls_history, 1):
                history_text += f"{i}. Herramienta '{call['tool']}':\n"
                history_text += f"   Input: {call['input']}\n"
                history_text += f"   Resultado: {call['result']}\n"
        
        full_prompt = f"{user_text}{history_text}"
        
        # Llamar al LLM (decisor)
        print(f"  📊 Consultando LLM (iteración {iteration_count})...")
        response = call_ollama_decision(full_prompt)
        
        print(f"  📝 LLM Response: {response}...")
        
        # Intentar parsear como JSON tool_call
        is_tool_call = False
        try:
            # Buscar JSON en la respuesta
            json_match = re.search(r'\{[^{}]*"type"\s*:\s*"tool_call"[^{}]*\}', response)
            
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
            else:
                response_json = json.loads(response.strip())
            
            if response_json.get("type") == "tool_call":
                is_tool_call = True
                tool_name = response_json.get("tool", "")
                tool_input = response_json.get("input", "")
                
                print(f"  🔧 Tool call detectado: '{tool_name}'")
                
                # Ejecutar herramienta
                tool_result = execute_tool(tool_name, tool_input)
                print(f"  ✅ Resultado: {tool_result}...")
                
                # Añadir al histórico
                tool_calls_history.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": tool_result
                })
                
                # Continuar el loop
                print(f"  🔄 Continuando loop (herramienta ejecutada)...")
                continue
        
        except json.JSONDecodeError as e:
            print(f"  ⚠️  No es JSON válido (asumiendo respuesta final): {e}")
        except Exception as e:
            print(f"  ⚠️  Error inesperado: {e}")
        
        # Si llegamos aquí: es respuesta final (no es tool_call)
        print(f"  ✨ Respuesta final generada por LLM")
        new_state = {
            **state,
            "reply_text": response,
            "tool_calls_history": tool_calls_history,
            "iteration_count": iteration_count
        }
        return new_state
    
    # Si llegamos al máximo de iteraciones
    print(f"  ⚠️  Límite de iteraciones alcanzado ({MAX_ITERATIONS})")
    print(f"  📝 Generando respuesta con información recopilada...")
    
    # Construir respuesta final con lo que se ha aprendido
    history_text = ""
    if tool_calls_history:
        history_text = "\nBasado en las siguientes herramientas ejecutadas:\n"
        for i, call in enumerate(tool_calls_history, 1):
            history_text += f"{i}. {call['tool']}: {call['result']}\n"
    
    final_prompt = f"Basado en: {user_text}{history_text}\n\nGenera una respuesta útil para el usuario."
    final_response = call_ollama_decision(final_prompt)
    
    return {
        **state,
        "reply_text": final_response,
        "tool_calls_history": tool_calls_history,
        "iteration_count": iteration_count
    }


def tts_node(state: GraphState) -> GraphState:
    """
    Nodo TTS (Text-To-Speech): Convierte texto a audio y lo reproduce.
    
    Entrada:
    - state["reply_text"]: str - Texto a reproducir
      Puede contener tags <think>...</think> de modelos de pensamiento
    
    Proceso:
    1. Obtener reply_text del estado
    2. FILTRADO: Eliminar bloques <think>...</think> (pensamiento interno del LLM)
    3. FILTRADO: Eliminar tags sueltos <think> o </think> (por si los hay)
    4. Pasar texto limpio a TTSEngine.speak()
    5. TTSEngine genera audio en memoria y lo reproduce
    
    Salida:
    - Reproduce audio (efecto secundario)
    - state["reply_text"]: str - Texto limpio (sin <think>)
    
    Retorna: Nuevo estado con reply_text limpio
    """
    # Obtener texto de respuesta (vacío por defecto si no existe)
    text = state.get("reply_text", "") or ""
    
    # Regex 1: Eliminar bloques <think>...</think> (DOTALL para saltos de línea)
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Regex 2: Eliminar tags sueltos <think>, </think>, <think/>, etc.
    cleaned = re.sub(
        r"</?think\s*/?>",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    
    # Limpiar espacios en blanco al inicio/final
    cleaned = cleaned.strip()
    
    # Reproducir audio limpio (sin <think> tags)
    tts.speak(cleaned)
    
    # Retornar estado con reply_text limpio
    return {**state, "reply_text": cleaned}


def build_graph():
    """
    Construye el grafo con ReAct Loop.
    
    Arquitectura SIMPLIFICADA:
    STT → Agent ReAct (loop interno) → TTS → END
    
    Ventajas:
    ✅ Un único agente maneja múltiples herramientas
    ✅ Loop iterativo interno (no requiere conditional_edges complejos)
    ✅ Histó rico de tools para context awareness
    ✅ Más fácil de debuggear
    ✅ Escalable para N herramientas
    
    Retorna: Grafo compilado
    
    Uso:
        app = build_graph()
        result = app.invoke({"frames": utterance_frames})
        # result["reply_text"] = respuesta final
        # result["tool_calls_history"] = [lista de tools ejecutadas]
        # Audio ya fue reproducido por tts_node
    """
    g = StateGraph(GraphState)
    
    # Registrar nodos
    g.add_node("stt", stt_node)
    g.add_node("agent_react", agent_react_node)
    g.add_node("tts", tts_node)

    # Definir entrada inicial del grafo
    g.set_entry_point("stt")
    
    # Definir aristas (flujo lineal y simple)
    g.add_edge("stt", "agent_react")      # STT → Agent ReAct
    g.add_edge("agent_react", "tts")      # Agent ReAct → TTS
    g.add_edge("tts", END)                # TTS → END
    
    # Compilar grafo
    return g.compile()
