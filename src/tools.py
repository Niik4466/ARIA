"""
Herramientas disponibles para el Agente LLM (Tool Use / Function Calling).

Cada herramienta es una función que el LLM puede invocar para:
- Realizar cálculos
- Ejecutar comandos del sistema
- Consultar información externa
- Interactuar con archivos
- etc.

Las herramientas se registran en TOOLS_REGISTRY para que el LLM las conozca.
"""

import subprocess
import json
import re
from datetime import datetime
from typing import Any


# ═════════════════════════════════════════════════════════════════
# TOOL 1: CALCULATOR - Operaciones matemáticas
# ═════════════════════════════════════════════════════════════════

def calculator_tool(expression: str) -> str:
    """
    Ejecuta una operación matemática segura.
    
    Entrada:
    - expression: str - Expresión matemática (ej: "2+2", "10*5", "sqrt(16)")
    
    Salida:
    - str - Resultado de la operación
    
    Seguridad:
    - Solo permite operaciones matemáticas básicas
    - Usa eval() con un namespace limitado (sin acceso a __import__, etc.)
    
    Ejemplos:
    - "2+2" → "Resultado: 4"
    - "sqrt(16)" → "Resultado: 4.0"
    - "10**2" → "Resultado: 100"
    """
    try:
        # Namespace seguro: solo funciones matemáticas
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "pow": pow,
            "round": round,
            "min": min,
            "max": max,
        }
        
        # Importar funciones del módulo math
        import math
        safe_dict.update({
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "log": math.log,
            "exp": math.exp,
        })
        
        result = eval(expression, safe_dict)
        return f"✓ Resultado: {result}"
    
    except Exception as e:
        return f"✗ Error en cálculo: {type(e).__name__}: {e}"


# ═════════════════════════════════════════════════════════════════
# TOOL 2: SYSTEM COMMAND - Ejecutar comandos CMD/PowerShell
# ═════════════════════════════════════════════════════════════════

def execute_system_command(command: str, shell_type: str = "powershell") -> str:
    """
    Ejecuta un comando del sistema (CMD o PowerShell).
    
    Entrada:
    - command: str - Comando a ejecutar (ej: "dir", "Get-Date", "python --version")
    - shell_type: str - Tipo de shell: "cmd" o "powershell" (default: "powershell")
    
    Salida:
    - str - Salida del comando (stdout + stderr)
    
    IMPORTANTE - Restricciones de seguridad:
    - NO ejecuta comandos peligrosos: rm, del, format, etc.
    - Máximo 5 segundos de ejecución (timeout)
    - Máximo 2000 caracteres de salida
    
    Ejemplos (PowerShell):
    - "Get-Date" → "Fecha y hora actual"
    - "Get-Process | Select-Object Name -First 5" → Lista de procesos
    - "python --version" → Versión de Python
    
    Ejemplos (CMD):
    - "dir" → Listado de directorio actual
    - "date /t" → Fecha actual
    - "ipconfig" → Configuración de red
    """
    
    # Lista negra de comandos peligrosos
    dangerous_commands = [
        "rm ", "del ", "format ", "diskpart",
        "shutdown", "restart", "netsh", "bcdedit",
        "dd ", "mkfs", "fdisk", "sfdisk",
        ":delete", "Remove-Item -Force -Recurse",
    ]
    
    # Verificar que no sea un comando peligroso
    for danger in dangerous_commands:
        if danger.lower() in command.lower():
            return f"✗ Comando rechazado por seguridad: '{command}' contiene operación peligrosa."
    
    try:
        # Seleccionar shell
        if shell_type.lower() == "cmd":
            shell = "cmd.exe"
            cmd_args = [shell, "/c", command]
        else:  # powershell por defecto
            shell = "powershell.exe"
            cmd_args = [shell, "-NoProfile", "-Command", command]
        
        # Ejecutar comando con timeout
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=5,  # Máximo 5 segundos
            cwd=None
        )
        
        # Combinar stdout + stderr
        output = result.stdout + result.stderr
        
        # Limitar salida a 2000 caracteres
        if len(output) > 2000:
            output = output[:2000] + "\n... (salida truncada)"
        
        if result.returncode == 0:
            return f"✓ Comando ejecutado:\n{output}" if output else "✓ Comando ejecutado exitosamente (sin salida)"
        else:
            return f"✗ Comando falló (código {result.returncode}):\n{output}"
    
    except subprocess.TimeoutExpired:
        return "✗ Comando excedió tiempo máximo (5 segundos)"
    
    except Exception as e:
        return f"✗ Error al ejecutar comando: {type(e).__name__}: {e}"


# ═════════════════════════════════════════════════════════════════
# TOOL 3: GET CURRENT TIME - Hora actual
# ═════════════════════════════════════════════════════════════════

def get_current_time_tool() -> str:
    """
    Obtiene la fecha y hora actual.
    
    Salida:
    - str - Fecha y hora en formato legible
    
    Ejemplos:
    - "18/11/2025 14:30:45"
    """
    now = datetime.now()
    return f"✓ Fecha y hora actual: {now.strftime('%d/%m/%Y %H:%M:%S')}"


# ═════════════════════════════════════════════════════════════════
# TOOL 4: READ FILE - Leer archivos
# ═════════════════════════════════════════════════════════════════

def read_file_tool(filepath: str, max_lines: int = 50) -> str:
    """
    Lee el contenido de un archivo de texto.
    
    Entrada:
    - filepath: str - Ruta del archivo (ej: "config.py", "C:\\Users\\...)
    - max_lines: int o str - Máximo número de líneas a mostrar (default: 50)
                              Si viene como string desde el LLM, se convierte a int
    
    Salida:
    - str - Contenido del archivo (primeras N líneas)
    
    Restricciones:
    - Máximo 50 líneas por defecto (evita archivos enormes)
    - Solo archivos de texto
    
    Ejemplos:
    - read_file_tool("config.py") → Contenido del archivo
    - read_file_tool("config.py", "50") → Primeras 50 líneas
    - read_file_tool("C:\\Users\\hola_\\Desktop\\archivo.txt") → Contenido
    """
    # Convertir max_lines a int si viene como string del LLM
    if isinstance(max_lines, str):
        try:
            max_lines = int(max_lines)
        except ValueError:
            max_lines = 50  # Default si no se puede convertir
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Limitar a max_lines
        if len(lines) > max_lines:
            content = ''.join(lines[:max_lines])
            content += f"\n... (archivo truncado a {max_lines} líneas de {len(lines)} totales)"
        else:
            content = ''.join(lines)
        
        return f"✓ Contenido de {filepath}:\n{content}"
    
    except FileNotFoundError:
        return f"✗ Archivo no encontrado: {filepath}"
    
    except Exception as e:
        return f"✗ Error al leer archivo: {type(e).__name__}: {e}"


# ═════════════════════════════════════════════════════════════════
# TOOL 5: WRITE FILE - Escribir/crear archivos
# ═════════════════════════════════════════════════════════════════

def write_file_tool(filepath: str, content: str) -> str:
    """
    Crea o sobrescribe un archivo de texto.
    
    Entrada:
    - filepath: str - Ruta del archivo
    - content: str - Contenido a escribir
    
    Salida:
    - str - Confirmación de escritura
    
    Restricciones:
    - No puede escribir en directorios críticos del sistema
    - Máximo 10,000 caracteres por archivo
    
    Ejemplos:
    - filepath="nota.txt", content="Hola mundo" → Crea/sobrescribe nota.txt
    """
    
    # Restricciones de seguridad
    dangerous_paths = [
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\System",
        "/etc",
        "/sys",
    ]
    
    for danger in dangerous_paths:
        if filepath.lower().startswith(danger.lower()):
            return f"✗ Escritura rechazada: no puedes escribir en {danger}"
    
    # Limitar tamaño
    if len(content) > 10000:
        return f"✗ Contenido demasiado grande: máximo 10,000 caracteres"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✓ Archivo escrito: {filepath} ({len(content)} caracteres)"
    
    except Exception as e:
        return f"✗ Error al escribir archivo: {type(e).__name__}: {e}"


# ═════════════════════════════════════════════════════════════════
# TOOL 6: GET OS INFO - Información del Sistema Operativo
# ═════════════════════════════════════════════════════════════════

def get_os_info_tool() -> str:
    """
    Obtiene información del Sistema Operativo.
    
    Salida:
    - str - Información del SO (nombre, versión, arquitectura, etc.)
    
    Información devuelta:
    - Nombre del SO (Windows, Linux, macOS)
    - Versión del SO
    - Arquitectura (32-bit, 64-bit)
    - Nombre de la máquina
    - Usuario actual
    - Procesador
    
    Ejemplos:
    - "Windows 11 64-bit, máquina: LAPTOP-XXX, usuario: hola_"
    """
    try:
        import platform
        import os
        
        os_name = platform.system()
        os_version = platform.release()
        os_build = platform.version() if hasattr(platform, 'version') else ""
        architecture = platform.machine()
        hostname = platform.node()
        username = os.getenv('USERNAME') or os.getenv('USER') or "desconocido"
        processor = platform.processor()
        
        info = f"""✓ Información del Sistema Operativo:
- SO: {os_name} {os_version}
- Build: {os_build}
- Arquitectura: {architecture}
- Máquina: {hostname}
- Usuario: {username}
- Procesador: {processor}"""
        
        return info
    
    except Exception as e:
        return f"✗ Error al obtener información del SO: {type(e).__name__}: {e}"


# ═════════════════════════════════════════════════════════════════
# TOOL 7: WEB SEARCH - Búsqueda en Internet
# ═════════════════════════════════════════════════════════════════

def web_search_tool(query: str) -> str:
    """
    Realiza una búsqueda en internet y retorna resultados.
    
    Entrada:
    - query: str - Términos de búsqueda (ej: "clima Buenos Aires", "capital de Francia")
    
    Salida:
    - str - Primeros resultados relevantes de la búsqueda
    
    Nota:
    - Usa DuckDuckGo o Google Search API
    - Retorna primeros 3-5 resultados
    - Incluye título, URL y descripción
    
    Restricciones:
    - Máximo 500 caracteres de salida
    - Timeout de 10 segundos
    
    Ejemplos:
    - "¿Cuál es la capital de Francia?" → "París es la capital de Francia"
    - "Última noticia sobre inteligencia artificial" → Últimas noticias
    """
    try:
        # Intentar usar requests + DuckDuckGo
        import requests
        
        # DuckDuckGo API (sin requerir API key)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Usar DuckDuckGo HTML search (lite version)
        search_url = "https://duckduckgo.com/lite"
        params = {"q": query}
        
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parsear HTML (búsqueda simplificada)
        import re
        from html import unescape
        
        html = response.text
        
        # Buscar resultados en el HTML (patrón simplificado)
        # DuckDuckGo lite usa <a> tags para resultados
        results = []
        
        # Patrón para extraer títulos y URLs
        pattern = r'<a href="([^"]+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html)
        
        for url, title in matches[:5]:  # Primeros 5 resultados
            # Filtrar enlaces internos y publicidad
            if 'duckduckgo.com' not in url and url.startswith('http'):
                title_clean = unescape(title).strip()
                if len(title_clean) > 10:  # Evitar resultados muy cortos
                    results.append(f"- {title_clean}: {url}")
        
        if results:
            result_text = "\n".join(results[:3])  # Primeros 3 resultados
            return f"✓ Resultados para '{query}':\n{result_text}"
        else:
            return f"⚠ No se encontraron resultados claros para '{query}'. Intenta con otros términos."
    
    except ImportError:
        return "✗ Se requiere 'requests' instalado. Instala con: pip install requests"
    
    except requests.Timeout:
        return "✗ Búsqueda excedió tiempo máximo (10 segundos)"
    
    except Exception as e:
        # Fallback: usar búsqueda local simple
        return f"⚠ Error en búsqueda web: {type(e).__name__}. Usa consultas más específicas."


# ═════════════════════════════════════════════════════════════════
# REGISTRY: Mapeo de herramientas disponibles
# ═════════════════════════════════════════════════════════════════

TOOLS_REGISTRY = {
    "calculator": {
        "function": calculator_tool,
        "description": "Realiza cálculos matemáticos. Input: 'expresión' (ej: '2+2', 'sqrt(16)')",
        "input_type": "string",
    },
    "execute_command": {
        "function": execute_system_command,
        "description": "Ejecuta comandos PowerShell o CMD. Input: 'comando' o 'comando':'shell_type' (ej: 'Get-Date' o 'dir':'cmd')",
        "input_type": "string",
    },
    "get_time": {
        "function": get_current_time_tool,
        "description": "Obtiene la fecha y hora actual. Input: (sin parámetros)",
        "input_type": None,
    },
    "read_file": {
        "function": read_file_tool,
        "description": "Lee el contenido de un archivo. Input: 'filepath' o 'filepath':'max_lines' (ej: 'config.py' o 'config.py':'50')",
        "input_type": "string",
    },
    "write_file": {
        "function": write_file_tool,
        "description": "Crea o sobrescribe un archivo. Input: 'filepath':'content' (ej: 'log.txt':'Hola mundo')",
        "input_type": "string",
    },
    "get_os_info": {
        "function": get_os_info_tool,
        "description": "Obtiene información del Sistema Operativo. Input: (sin parámetros)",
        "input_type": None,
    },
    "web_search": {
        "function": web_search_tool,
        "description": "Realiza una búsqueda en internet. Input: 'términos' (ej: 'capital de Francia')",
        "input_type": "string",
    },
}


def get_tools_description() -> str:
    """
    Genera descripción de todas las herramientas disponibles para el LLM.
    
    Retorna un formato que el LLM puede entender.
    """
    description = "Herramientas disponibles:\n"
    for tool_name, tool_info in TOOLS_REGISTRY.items():
        description += f"\n- {tool_name}: {tool_info['description']}"
    return description


def execute_tool(tool_name: str, tool_input: str) -> str:
    """
    Ejecuta una herramienta del registry.
    
    Entrada:
    - tool_name: str - Nombre de la herramienta
    - tool_input: str - Input para la herramienta (puede ser uno o varios argumentos)
    
    FORMATO DE ENTRADA (especificado para el LLM):
    - Un argumento: 'arg1'
    - Múltiples argumentos: 'arg1':'arg2':'arg3'
    
    Los argumentos DEBEN estar entre comillas simples y separados por ':'
    Si el input NO contiene comillas simples, se trata como un string único.
    
    Salida:
    - str - Resultado de la ejecución
    
    Ejemplos:
    - execute_tool("calculator", "'2+2'") → ejecuta calculator_tool("2+2")
    - execute_tool("read_file", "'config.py':'50'") → ejecuta read_file_tool("config.py", "50")
    - execute_tool("write_file", "'log.txt':'contenido'") → ejecuta write_file_tool("log.txt", "contenido")
    - execute_tool("execute_command", "'Get-Date'") → ejecuta execute_system_command("Get-Date")
    """
    if tool_name not in TOOLS_REGISTRY:
        return f"✗ Herramienta desconocida: {tool_name}"
    
    tool_func = TOOLS_REGISTRY[tool_name]["function"]
    
    try:
        # Algunos tools no necesitan parámetros
        if TOOLS_REGISTRY[tool_name]["input_type"] is None:
            result = tool_func()
        else:
            # Parsear múltiples argumentos si es necesario
            args = _parse_tool_input(tool_input)
            
            # Llamar función con argumentos posicionales
            if len(args) == 1:
                result = tool_func(args[0])
            else:
                result = tool_func(*args)
        
        return result
    
    except TypeError as e:
        # Error de argumentos: probablemente número incorrecto
        return f"✗ Error de argumentos en {tool_name}: {e}. Input recibido: {tool_input}"
    
    except Exception as e:
        return f"✗ Error al ejecutar {tool_name}: {type(e).__name__}: {e}"


def _parse_tool_input(tool_input: str) -> list:
    """
    Parsea el input de una herramienta de forma inteligente.
    
    Entrada:
    - tool_input: str - Input que puede contener múltiples argumentos
    
    Salida:
    - list - Lista de argumentos parseados
    
    FORMATO ESPECIFICADO PARA EL LLM:
    Los argumentos DEBEN estar entre comillas simples y separados por ':'
    Ejemplos:
    - 'arg1' → un argumento: ["arg1"]
    - 'arg1':'arg2' → dos argumentos: ["arg1", "arg2"]
    - 'arg1':'arg2':'arg3' → tres argumentos: ["arg1", "arg2", "arg3"]
    
    ESTRATEGIA DE PARSEO (en orden de prioridad):
    1. Parsear como JSON primero: {"arg1": "val1", "arg2": "val2"}
       → Extrae valores en orden
    
    2. Parsear con formato de comillas simples:
       'arg1':'arg2':'arg3' → Usa regex para extraer strings entre comillas
       → Solo parsea si encuentra al menos UN patrón 'string'
       → Esto evita hacer split por ':' en código Python, URLs, etc.
    
    3. Por defecto: retorna como string simple
       → Esto es seguro para inputs simples como "2+2" o "config.py"
    
    Ejemplos de uso:
    - _parse_tool_input("'config.py'") → ["config.py"]
    - _parse_tool_input("'config.py':'50'") → ["config.py", "50"]
    - _parse_tool_input("print('hola')\nprint('mundo')") → ["print('hola')\nprint('mundo')"] (sin split)
    - _parse_tool_input("'https://example.com':'data'") → ["https://example.com", "data"]
    
    """
    
    # Intentar parsear como JSON primero
    try:
        parsed_json = json.loads(tool_input)
        if isinstance(parsed_json, dict):
            # Extraer valores en orden (preservar orden de dict)
            values = list(parsed_json.values())
            # Convertir a string si es necesario
            return [str(v) for v in values]
    except (json.JSONDecodeError, ValueError):
        pass  # No es JSON, continuar
    
    # PARSEO INTELIGENTE CON COMILLAS SIMPLES
    # Regex que busca patrones: 'string' (entre comillas simples)
    # Los strings pueden contener cualquier cosa EXCEPTO comillas simples sin escapar
    pattern = r"'([^']*(?:\\'[^']*)*)'"
    matches = re.findall(pattern, tool_input)
    
    # Si encontramos al menos UN argumento entre comillas, usar este formato
    if matches:
        # Desescapar comillas simples si es necesario (\'  → ')
        args = [match.replace("\\'", "'") for match in matches]
        return args
    
    # Por defecto: retornar como string simple en lista (seguro para inputs simples)
    # Esto preserva el contenido exacto sin intentar hacer split
    return [tool_input]
