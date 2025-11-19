"""
EJEMPLOS de uso del sistema con Tool Use.

Este archivo muestra ejemplos de cómo el LLM invoca herramientas externas.
NO ejecutar este archivo; es solo para referencia.
"""

# ═════════════════════════════════════════════════════════════════
# EJEMPLO 1: Preguntar la hora
# ═════════════════════════════════════════════════════════════════

User dice:
"¿Qué hora es?"

Flujo:
1. STT: "¿Qué hora es?" → user_text
2. Agent: LLM decide que necesita "get_time"
   Respuesta LLM: {"type": "tool_call", "tool": "get_time", "input": ""}
3. Tool executor: Ejecuta get_current_time_tool()
   Resultado: "✓ Fecha y hora actual: 18/11/2025 14:30:45"
4. Agent: Reenvía resultado a LLM
   Final LLM: "Son las 14:30 y 45 segundos"
5. TTS: Reproduce "Son las 14:30 y 45 segundos"

---

# ═════════════════════════════════════════════════════════════════
# EJEMPLO 2: Cálculo matemático
# ═════════════════════════════════════════════════════════════════

User dice:
"¿Cuánto es 1234 por 567?"

Flujo:
1. STT: "¿Cuánto es 1234 por 567?" → user_text
2. Agent: LLM decide que necesita "calculator"
   Respuesta LLM: {"type": "tool_call", "tool": "calculator", "input": "1234*567"}
3. Tool executor: Ejecuta calculator_tool("1234*567")
   Resultado: "✓ Resultado: 699678"
4. Agent: Reenvía resultado a LLM
   Final LLM: "El resultado de 1234 por 567 es 699678"
5. TTS: Reproduce "El resultado de 1234 por 567 es 699678"

---

# ═════════════════════════════════════════════════════════════════
# EJEMPLO 3: Ejecutar comando del sistema
# ═════════════════════════════════════════════════════════════════

User dice:
"¿Qué versión de Python tengo instalada?"

Flujo:
1. STT: "¿Qué versión de Python tengo instalada?" → user_text
2. Agent: LLM decide que necesita "execute_command"
   Respuesta LLM: {
       "type": "tool_call",
       "tool": "execute_command",
       "input": "python --version"
   }
3. Tool executor: Ejecuta execute_system_command("python --version", "powershell")
   Resultado: "✓ Comando ejecutado:\nPython 3.11.4"
4. Agent: Reenvía resultado a LLM
   Final LLM: "Tienes Python versión 3.11.4 instalado"
5. TTS: Reproduce "Tienes Python versión 3.11.4 instalado"

---

# ═════════════════════════════════════════════════════════════════
# EJEMPLO 4: Leer un archivo
# ═════════════════════════════════════════════════════════════════

User dice:
"Lee el archivo config.py"

Flujo:
1. STT: "Lee el archivo config.py" → user_text
2. Agent: LLM decide que necesita "read_file"
   Respuesta LLM: {
       "type": "tool_call",
       "tool": "read_file",
       "input": "config.py"
   }
3. Tool executor: Ejecuta read_file_tool("config.py")
   Resultado: "✓ Contenido de config.py:\n[primeras 50 líneas del archivo]"
4. Agent: Reenvía resultado a LLM
   Final LLM: "El archivo config.py contiene la configuración del proyecto. 
              Define SAMPLE_RATE = 16000, ..."
5. TTS: Reproduce resumen del archivo

---

# ═════════════════════════════════════════════════════════════════
# EJEMPLO 5: Respuesta directa (sin tools)
# ═════════════════════════════════════════════════════════════════

User dice:
"Hola, ¿cómo estás?"

Flujo:
1. STT: "Hola, ¿cómo estás?" → user_text
2. Agent: LLM analiza y decide que NO necesita herramientas
   Respuesta LLM: "Estoy bien, gracias por preguntar. ¿Cómo puedo ayudarte?"
   (Texto normal, no JSON)
3. Agent: Detecta que es texto normal (no JSON)
   Retorna directamente: "Estoy bien, gracias por preguntar. ¿Cómo puedo ayudarte?"
4. TTS: Reproduce respuesta

---

# ═════════════════════════════════════════════════════════════════
# HERRAMIENTAS DISPONIBLES
# ═════════════════════════════════════════════════════════════════

1. calculator
   - Input: Expresión matemática (ej: "2+2", "sqrt(16)", "10**2")
   - Output: Resultado del cálculo
   - Ejemplo: "¿Cuánto es 2+2?" → 4

2. execute_command
   - Input: Comando PowerShell/CMD (ej: "Get-Date", "dir", "python --version")
   - Output: Salida del comando
   - Ejemplo: "¿Qué hora es?" → [se ejecuta Get-Date]

3. get_time
   - Input: (sin parámetros)
   - Output: Fecha y hora actual
   - Ejemplo: "¿Qué hora es?" → [se invoca automáticamente]

4. read_file
   - Input: Ruta del archivo (ej: "config.py", "C:\\Users\\hola_\\Desktop\\archivo.txt")
   - Output: Contenido del archivo (primeras 50 líneas)
   - Ejemplo: "Lee config.py" → [contenido del archivo]

5. write_file
   - Input: "ruta:contenido" (ej: "nota.txt:Hola mundo")
   - Output: Confirmación de escritura
   - Ejemplo: "Crea un archivo llamado log.txt con el texto: Error encontrado"
   → [se crea log.txt con contenido]

---

# ═════════════════════════════════════════════════════════════════
# CÓMO AGREGAR NUEVAS HERRAMIENTAS
# ═════════════════════════════════════════════════════════════════

1. Crear función en src/tools.py

   def my_tool(input_param: str) -> str:
       \"\"\"Descripción de la herramienta.\"\"\"
       # Implementar lógica
       return resultado

2. Registrar en TOOLS_REGISTRY

   TOOLS_REGISTRY = {
       "my_tool": {
           "function": my_tool,
           "description": "Descripción breve",
           "input_type": "string",
       },
       # ... resto de tools
   }

3. ¡Listo! El LLM ya puede invocar tu herramienta automáticamente

---

# ═════════════════════════════════════════════════════════════════
# NOTAS IMPORTANTES
# ═════════════════════════════════════════════════════════════════

- Las herramientas tienen restricciones de seguridad integradas
  * No permite comandos peligrosos (rm, del, format, etc.)
  * Máximo de caracteres de salida (2000)
  * Timeout de 5 segundos para comandos

- El LLM decide automáticamente si necesita una herramienta
  * Si responde con JSON → se ejecuta la tool
  * Si responde con texto → respuesta directa

- Cada invocación de tool consume recursos
  * El LLM reinvoca después de recibir el resultado
  * Total de latencia: STT + LLM + Tool + LLM final + TTS

- Error handling
  * Si una tool falla, el LLM recibe el error y puede reintentar
  * Si el JSON es malformado, se trata como texto normal
"""
