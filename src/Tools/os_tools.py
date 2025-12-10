
import subprocess
import os
import platform
import ctypes

def execute_system_command(command: str, shell_type: str = "powershell") -> str:
    """
    Ejecuta un comando del sistema (CMD o PowerShell).
    """
    # Lista negra de comandos peligrosos
    dangerous_commands = [
        "rm ", "del ", "format ", "diskpart",
        "shutdown", "restart", "netsh", "bcdedit",
        "dd ", "mkfs", "fdisk", "sfdisk",
        ":delete", "Remove-Item -Force -Recurse",
    ]
    
    for danger in dangerous_commands:
        if danger.lower() in command.lower():
            return f"✗ Comando rechazado por seguridad: '{command}' contiene operación peligrosa."
    
    try:
        if shell_type.lower() == "cmd":
            shell = "cmd.exe"
            cmd_args = [shell, "/c", command]
        else:  # powershell por defecto
            shell = "powershell.exe"
            cmd_args = [shell, "-NoProfile", "-Command", command]
        
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=None
        )
        
        output = result.stdout + result.stderr
        
        if len(output) > 2000:
            output = output[:2000] + "\n... (salida truncada)"
        
        if result.returncode == 0:
            return f"✓ Comando {command} ejecutado:\n{output}" if output else "✓ Comando {command} ejecutado exitosamente"
        else:
            return f"✗ Comando {command} falló (código {result.returncode}):\n{output}"
    
    except subprocess.TimeoutExpired:
        return "✗ Comando excedió tiempo máximo (5 segundos)"
    
    except Exception as e:
        return f"✗ Error al ejecutar comando: {type(e).__name__}: {e}"

def read_file_tool(filepath: str, max_lines: int = 50) -> str:
    """
    Lee el contenido de un archivo de texto.
    """
    if isinstance(max_lines, str):
        try:
            max_lines = int(max_lines)
        except ValueError:
            max_lines = 50
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
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

def write_file_tool(filepath: str, content: str) -> str:
    """
    Crea o sobrescribe un archivo de texto.
    """
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
    
    if len(content) > 10000:
        return f"✗ Contenido demasiado grande: máximo 10,000 caracteres"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✓ Archivo escrito con éxito en {filepath}, ({len(content)} caracteres escritos)"
    
    except Exception as e:
        return f"✗ Error al escribir archivo: {type(e).__name__}: {e}"

def get_os_info_tool() -> str:
    """
    Obtiene información del Sistema Operativo.
    """
    try:
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

def media_play_pause_tool() -> str:
    """
    Envía UNA SOLA VEZ la tecla multimedia Play/Pause al sistema (Windows).
    """
    try:
        if platform.system() != "Windows":
            return (
                "✗ media_play_pause_tool: esta herramienta solo funciona en Windows. "
                "No vuelvas a llamarla para esta petición si el sistema no es Windows."
            )
        
        # Constantes de Windows
        VK_MEDIA_PLAY_PAUSE = 0xB3
        KEYEVENTF_EXTENDEDKEY = 0x0001
        KEYEVENTF_KEYUP       = 0x0002

        user32 = ctypes.WinDLL("user32", use_last_error=True)

        user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY, 0)
        user32.keybd_event(
            VK_MEDIA_PLAY_PAUSE,
            0,
            KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP,
            0
        )

        return (
            "✓ ACCIÓN REALIZADA: Se envió UNA sola vez la tecla multimedia Play/Pause "
            "al sistema para alternar el estado de reproducción.\n"
            "IMPORTANTE (para el agente): No vuelvas a llamar media_play_pause_tool "
        )
    
    except Exception as e:
        return (
            "✗ Error al enviar tecla Play/Pause: "
            f"{type(e).__name__}: {e}. "
            "No intentes repetir esta herramienta en bucle; informa el error al usuario."
        )
