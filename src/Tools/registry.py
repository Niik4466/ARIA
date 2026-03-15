
import json
import re
from .basic_tools import calculator_tool, get_current_time_tool
from .os_tools import execute_system_command, read_file_tool, write_file_tool, get_os_info_tool, media_play_pause_tool
from .search_tools import web_search_tool
from .autoconfig import add_wakeword_samples_tool, new_wakeword_tool, change_voice_tool, regenerate_voice_tool

TOOLS_REGISTRY = {
    "calculator": {
        "function": calculator_tool,
        "description": "Realiza cálculos matemáticos. arg1:'expresión' (ej: arg1:'2+2' o arg1:'sqrt(16)')",
        "category": "basic"
    },
    "get_time": {
        "function": get_current_time_tool,
        "description": "Obtiene la fecha y hora actual. args: (sin argumentos)",
        "category": "basic"
    },
    "execute_command": {
        "function": execute_system_command,
        "description": "Ejecuta comandos PowerShell o CMD. arg1:'comando' o arg1:'comando', arg2:'shell_type' (ej: arg1:'Get-Date' o arg1:'dir', arg2:'cmd')",
        "category": "os"
    },
    "read_file": {
        "function": read_file_tool,
        "description": "Lee el contenido de un archivo. arg1:'filepath', arg2:'max_lines' (ej: arg1:'config.py' o arg1:'config.py', arg2:'50')",
        "category": "os"
    },
    "write_file": {
        "function": write_file_tool,
        "description": "Crea o sobrescribe un archivo. arg1:'filepath', arg2:'content' (ej: arg1:'log.txt', arg2:'Hola mundo')",
        "category": "os"
    },
    "get_os_info": {
        "function": get_os_info_tool,
        "description": "Obtiene información del Sistema Operativo. args: (sin argumentos)",
        "category": "os"
    },
    "media_play_pause": {
        "function": media_play_pause_tool,
        "description": "Alterna el estado de reproducción del audio del sistema. args: (sin argumentos)",
        "category": "os"
    },
    "web_search": {
        "function": web_search_tool,
        "description": "Busca información actualizada en internet. arg1:'query' (ej: arg1:'noticias de hoy' o arg1:'python tutorial')",
        "category": "search"
    },
    "add_wakeword_samples": {
        "function": add_wakeword_samples_tool,
        "description": "Agrega muestras extras para mejorar el reconocimiento de la palabra de activación. arg1:'num_samples' (ej: arg1:'2')",
        "category": "autoconfig"
    },
    "new_wakeword": {
        "function": new_wakeword_tool,
        "description": "Borra la wakeword actual y crea una nueva palabra de activación guiando al usuario. args: (sin argumentos)",
        "category": "autoconfig"
    },
    "change_voice": {
        "function": change_voice_tool,
        "description": "Changes the assistant's voice based on a textual description. arg1:'text' (text to say), arg2:'instruct' (voice info e.g. 'Female, 24 years old'), arg3:'language' (e.g. 'English')",
        "category": "autoconfig"
    },
    "regenerate_voice": {
        "function": regenerate_voice_tool,
        "description": "Regenerates the assistant's voice using previously stored metadata. args: (no arguments)",
        "category": "autoconfig"
    }
}

def get_tools_description(category: str = None) -> str:
    """
    Genera descripción de herramientas disponibles.
    Si se especifica category, solo devuelve herramientas de esa categoría.
    """
    description = "Herramientas disponibles:\n"
    for tool_name, tool_info in TOOLS_REGISTRY.items():
        if category and tool_info.get("category") != category:
            continue
        description += f"\n- {tool_name}: {tool_info['description']}"
    return description

def _parse_tool_input(tool_input: str) -> list:
    """
    Parsea el input de una herramienta.
    """
    # Intentar parsear como JSON primero (nuevo formato)
    if isinstance(tool_input, dict):
        return list(tool_input.values())

    try:
        parsed_json = json.loads(tool_input)
        if isinstance(parsed_json, dict):
            return list(parsed_json.values())
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Intento de parseo antiguo (compatibility fallback) pero el nuevo prompt pide JSON
    # Aún así mantenemos lógica robusta
    return [str(tool_input)]

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Ejecuta una herramienta del registry.
    tool_args espera ser un dict con los argumentos, ej: {"arg1": "val1"}
    """
    if tool_name not in TOOLS_REGISTRY:
        return f"✗ Herramienta desconocida: {tool_name}"
    
    tool_func = TOOLS_REGISTRY[tool_name]["function"]
    
    try:
        # Extraer valores de los argumentos en orden
        args = list(tool_args.values())
        
        # Llamar función
        if len(args) == 0:
            result = tool_func()
        elif len(args) == 1:
            result = tool_func(args[0])
        else:
            result = tool_func(*args)
        
        return result
    
    except Exception as e:
        return f"✗ Error al ejecutar {tool_name}: {type(e).__name__}: {e}"
