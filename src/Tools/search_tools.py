from ddgs import DDGS
import time

from config import verbose_mode
_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)

def web_search_tool(query: str) -> str:
    """
    Realiza una búsqueda usando duckduckgo-search. 
    Intenta múltiples backends si el primero falla.
    """
    
    # Intentamos primero con el backend 'api' (más rápido), si falla, 'html' (más robusto)
    backends = ['html', 'api', 'lite']
    
    for backend in backends:
        try:
            with DDGS() as ddgs:
                results_gen = ddgs.text(
                    query=query, 
                    region="wt-wt", 
                    safesearch="off", 
                    timelimit="d",
                    max_results=4,
                    backend=backend
                )
                
                results = []
                # Convertimos el generador a lista para verificar si está vacío
                for r in results_gen:
                    title = r.get('title', 'Sin título')
                    href = r.get('href', 'No URL')
                    body = r.get('body', '')
                    results.append(f"- **{title}**\n  URL: {href}\n  Resumen: {body}")
                
                if results:
                    result_text = "\n\n".join(results)
                    return f"✓ Resultados ({backend}) para '{query}':\n\n{result_text}"
                
        except Exception as e:
            print(f"⚠ Fallo backend '{backend}': {e}")
            time.sleep(1) # Pequeña pausa antes de reintentar
            continue

    return f"⚠ No se encontraron resultados para '{query}' tras intentar múltiples métodos."