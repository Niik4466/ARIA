
import math
from datetime import datetime

def calculator_tool(expression: str) -> str:
    """
    Ejecuta una operación matemática segura.
    
    Entrada:
    - expression: str - Expresión matemática (ej: "2+2", "10*5", "sqrt(16)")
    
    Salida:
    - str - Resultado de la operación
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

def get_current_time_tool() -> str:
    """
    Obtiene la fecha y hora actual.
    
    Salida:
    - str - Fecha y hora en formato legible
    """
    now = datetime.now()
    return f"✓ Fecha y hora actual: {now.strftime('%d/%m/%Y %H:%M:%S')}"
