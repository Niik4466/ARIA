"""
Módulo de Text-To-Speech (TTS) usando edge-tts.

Propósito: Convertir texto a audio y reproducirlo directamente en memoria 
sin escribir archivos temporales en disco.

Características clave:
- Genera audio en memoria usando edge-tts (Microsoft Azure TTS online)
- Decodifica audio en memory (no en disco)
- Reproduce audio en tiempo real con sounddevice
- Fallback con ffmpeg si libsndfile no soporta el formato

Flujo de datos:
1. Text input (string en español)
2. edge_tts.Communicate.stream() genera chunks de audio
3. Acumular chunks en bytearray (memoria)
4. Decodificar bytes a PCM float32
5. Reproducir con sounddevice.play()

"""

import asyncio
import base64
import io
import shutil
import edge_tts
import sounddevice as sd
import soundfile as sf
from config import VOICE_NAME


class TTSEngine:
    """
    Motor TTS que usa edge-tts (Microsoft Azure Text-to-Speech).
    
    Características:
    - Soporta múltiples voces (idiomas y acentos)
    - Genera audio MP3 por defecto
    - Decodificación en memoria (sin archivos temporales)
    - Reproducción automática con sounddevice
    
    Atributos:
    - voice: str - ID de voz (ej. "es-ES-AlvaroNeural", "en-US-AriaNeural")
             Configurable en config.VOICE_NAME
    """
    
    def __init__(self):
        """
        Inicializa el motor TTS.
        
        Atributo:
        - self.voice: Voz configurada en config.py
                     Ej. "es-ES-AlvaroNeural" para español de España, voz masculina
        """
        self.voice = VOICE_NAME

    async def speak_async(self, text: str):
        """
        Genera audio desde texto y lo reproduce (versión async).
        
        Parámetro:
        - text: str - Texto a reproducir en audio
        
        Proceso:
        1. Si text está vacío, retornar (no hacer nada)
        2. Crear Communicate con edge-tts
        3. Usar communicate.stream() para obtener chunks de audio
        4. Acumular chunks en bytearray (memoria)
        5. Decodificar bytes a PCM float32
        6. Reproducir con sounddevice
        
        Detalles:
        - edge_tts.Communicate entrega audio en chunks (streaming)
        - Cada chunk viene en mensaje dict con "type": "audio"
        - Audio puede estar en base64 o bytes crudo
        - Acumulamos todo en bytearray (típicamente 100KB-1MB)
        - Luego decodificamos y reproducimos
        
        Fallback: Si libsndfile no soporta el formato (ej. MP3),
        intentamos ffmpeg para decodificar en memory.
        """
        if not text:
            return

        # Crear comunicador edge-tts
        # "rate="+0%" = velocidad normal (0% cambio)
        # "voice=..." = voz específica (ej. "es-ES-AlvaroNeural")
        communicate = edge_tts.Communicate(text, voice=self.voice, rate="+0%")

        # ════════════════════════════════════════════════════════════════
        # Fase 1: Recolectar chunks de audio en memoria (bytearray)
        # ════════════════════════════════════════════════════════════════
        audio_buf = bytearray()
        
        # Usar stream() para obtener chunks de audio asincronamente
        # Típicamente entrega mensaje por cada chunk:
        # {"type": "audio", "data": "<base64 encoded MP3>" o bytes}
        async for msg in communicate.stream():
            if msg.get("type") == "audio":
                # Extraer audio del mensaje
                # Puede venir como "data" o "audio", o como bytes o base64 string
                data = msg.get("data") or msg.get("audio")
                if data is None:
                    continue
                
                # Si es string, probablemente es base64
                if isinstance(data, str):
                    try:
                        chunk = base64.b64decode(data)
                    except Exception:
                        # No es base64 válido, ignorar
                        continue
                else:
                    # Ya son bytes
                    chunk = data
                
                # Añadir chunk al buffer de memoria
                audio_buf.extend(chunk)

        if not audio_buf:
            # No hay audio, salir
            return

        # ════════════════════════════════════════════════════════════════
        # Fase 2: Decodificar audio en memoria
        # ════════════════════════════════════════════════════════════════
        bio = io.BytesIO(audio_buf)

        # Intento 1: Usar soundfile (requiere libsndfile con soporte del formato)
        try:
            # soundfile.read() puede decodificar WAV, OGG, FLAC, etc.
            # Si edge-tts entrega MP3, podría fallar aquí
            data, samplerate = sf.read(bio, dtype="float32")
            sd.play(data, samplerate)
            sd.wait()
            print(f"🔊 TTS reproducido en memoria (samplerate={samplerate})")
            return
        except Exception as e:
            # Fallback: intentar ffmpeg si está disponible
            if shutil.which("ffmpeg"):
                try:
                    # Usar ffmpeg para decodificar en memory
                    # ffmpeg: pipe:0 = stdin, pipe:1 = stdout
                    # Entrada: MP3/etc del audio_buf
                    # Salida: WAV (para que soundfile pueda decodificar)
                    proc = await asyncio.create_subprocess_exec(
                        "ffmpeg",
                        "-hide_banner",       # No mostrar banner
                        "-loglevel", "error", # Solo errores
                        "-i", "pipe:0",       # Entrada desde stdin
                        "-f", "wav",          # Salida en formato WAV
                        "pipe:1",             # Salida a stdout
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    
                    # Ejecutar ffmpeg con input/output
                    out, err = await proc.communicate(bytes(audio_buf))
                    
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"ffmpeg failed to decode audio: {err.decode(errors='ignore')}"
                        )
                    
                    # Decodificar WAV generado por ffmpeg
                    bio2 = io.BytesIO(out)
                    data, samplerate = sf.read(bio2, dtype="float32")
                    sd.play(data, samplerate)
                    sd.wait()
                    print("🔊 TTS reproducido en memoria (ffmpeg fallback)")
                    return
                except Exception as ffmpeg_err:
                    raise RuntimeError(
                        f"Failed to decode audio with ffmpeg: {ffmpeg_err}"
                    )
            
            # Si llegamos aquí, no pudimos decodificar
            raise RuntimeError(
                "No se pudo decodificar el audio en memoria. "
                "Instala ffmpeg o habilita soporte MP3 en libsndfile. "
                f"Original error: {e}"
            )

    def speak(self, text: str):
        """
        Genera audio desde texto y lo reproduce (versión síncrona).
        
        Parámetro:
        - text: str - Texto a reproducir
        
        Nota: Envuelve speak_async() en asyncio.run() para uso síncrono
        desde código no-async (como main.py y graph.py).
        
        Bloquea hasta que el audio termine de reproducirse.
        """
        asyncio.run(self.speak_async(text))
