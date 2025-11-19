"""
Módulo de Captura de Audio desde el Micrófono.

Propósito: Capturar audio en tiempo real usando sounddevice y organizarlo en frames
para procesamiento posterior (VAD, STT, etc.).

Flujo de datos:
1. sounddevice captura audio del micrófono en tiempo real
2. El callback _callback recibe bloques de audio
3. Se organiza en frames fijos de FRAME_MS (20ms)
4. Frames se encolan solo si está activo listening
5. Consumidor (VAD) extrae frames mediante iterator .frames()
"""

import queue
import threading
import sounddevice as sd
import numpy as np
from typing import Iterator, Optional
from config import SAMPLE_RATE, CHANNELS, FRAME_MS

# Calcular tamaño de frame en muestras (samples)
# SAMPLE_RATE=16000 Hz, FRAME_MS=20ms => 16000*20/1000 = 320 samples
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


class MicrophoneStream:
    """
    Capturador de audio del micrófono con control manual de escucha.
    
    Características:
    - Usa sounddevice.InputStream para captura PCM en tiempo real
    - Organiza audio en frames de FRAME_MS ms
    - Soporta start_listening() / stop_listening() para control manual
    - Thread-safe: usa locks para sincronización
    
    Flujo de datos:
    - sounddevice → callback _callback → queue → iterator .frames()
    
    Estados:
    - listening=False: No encola frames (están cayendo)
    - listening=True: Encola frames en la queue
    """
    
    def __init__(self):
        """
        Inicializa el stream de micrófono (pero no lo inicia aún).
        
        Atributos:
        - q: Queue thread-safe que almacena frames numpy int16
        - stream: El objeto InputStream de sounddevice (None hasta __enter__)
        - listening: Bandera para controlar si encolar frames
        - listening_lock: Lock para sincronizar acceso a 'listening'
        """
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = None
        self.listening = False  # Control manual de escucha
        self.listening_lock = threading.Lock()

    def _callback(self, indata, frames, time, status):
        """
        Callback invocado por sounddevice cada vez que hay nuevo audio disponible.
        
        Parámetros:
        - indata: Array de audio del micrófono (shape=(N, CHANNELS))
        - frames: Número de frames que sounddevice entregó
        - time: Información de timing
        - status: Flags de error (nonzero si hay problemas)
        
        Proceso:
        1. Si hay error, imprimimos (pero continuamos)
        2. Convertir float32 a int16 PCM (rango -1.0 a 1.0 → -32767 a 32767)
        3. Extraer canal 0 (mono)
        4. Dividir en chunks de FRAME_SAMPLES
        5. Si estamos listening, encolar el frame
        
        Importante: Este callback se ejecuta en thread de audio, debe ser rápido.
        """
        if status:
            print("⚠️  audio status:", status)
        
        # Convertir a int16 mono
        # sounddevice entrega float32 por defecto, convertir a int16 (rango PCM)
        if indata.dtype != np.int16:
            pcm = (indata[:, 0] * 32767).astype(np.int16)
        else:
            pcm = indata[:, 0]
        
        # Dividir en frames de FRAME_SAMPLES
        # Por ej. si sounddevice entrega 480 samples y FRAME_SAMPLES=320,
        # dividimos en chunks de 320 (un chunk completo)
        for start in range(0, len(pcm), FRAME_SAMPLES):
            chunk = pcm[start:start+FRAME_SAMPLES]
            # Solo encolar si es un frame completo (no fragmentos)
            if len(chunk) == FRAME_SAMPLES:
                with self.listening_lock:
                    # Solo encolar si estamos escuchando
                    if self.listening:
                        self.q.put(chunk.copy())

    def __enter__(self):
        """
        Context manager: Inicia el stream de micrófono.
        
        Crea un InputStream de sounddevice:
        - channels=CHANNELS (mono, 1 canal)
        - samplerate=16000 Hz (requerido por WebRTC VAD)
        - dtype='int16' (solicitar int16 nativamente si es posible)
        - callback=_callback (invocar cada vez que hay audio)
        - blocksize=FRAME_SAMPLES (hint a sounddevice de tamaño de bloque)
        
        Retorna: self (para usar en 'with MicrophoneStream() as mic:')
        """
        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype='int16',
            callback=self._callback,
            blocksize=FRAME_SAMPLES
        )
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        Context manager: Detiene el stream de micrófono.
        
        Se llama automáticamente al salir del bloque 'with'.
        Detiene y cierra el stream de sounddevice.
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def start_listening(self):
        """
        Activa la captura de audio.
        
        A partir de este momento, _callback encolará frames en la queue.
        Se usa cuando el usuario presiona Ctrl+Alt+Espacio.
        """
        with self.listening_lock:
            self.listening = True

    def stop_listening(self):
        """
        Detiene la captura de audio.
        
        A partir de este momento, _callback descartará frames (no encolará).
        Se usa cuando el usuario suelta las teclas.
        
        También limpia la cola de frames pendientes para no procesar audio antiguo.
        """
        with self.listening_lock:
            self.listening = False
        # No vaciamos la cola aquí: el consumidor debe llamar a drain_frames()
        # para obtener los frames capturados durante la sesión de escucha.

    def drain_frames(self) -> list:
        """
        Vacía la cola de frames y devuelve una lista con todos los frames
        capturados hasta el momento.

        Esto permite a quien controla la lógica (por ejemplo main.py)
        obtener de forma atómica todos los frames que se capturaron
        durante la sesión de escucha, y procesarlos cuando sea necesario.
        """
        frames = []
        # Vaciar la cola rápidamente
        while not self.q.empty():
            try:
                frames.append(self.q.get_nowait())
            except queue.Empty:
                break
        return frames

    def frames(self, timeout: Optional[float] = None) -> Iterator[np.ndarray]:
        """
        Generador que produce frames de audio mientras está activa la escucha.
        
        Parámetros:
        - timeout: Tiempo máximo de espera por frame (segundos). Si None, espera 1.0s.
        
        Rendimiento: Yields frames numpy int16 de tamaño FRAME_SAMPLES.
        
        Comportamiento:
        - Mientras listening=True, intenta obtener frames de la queue
        - Si timeout se agota y listening=False, termina el generador
        - Si timeout se agota pero listening=True, reintentar (esperar más)
        
        Uso típico:
            for frame in mic.frames():
                process(frame)  # Se ejecuta para cada frame
                if condition:
                    break  # O mic.stop_listening() interrumpe
        """
        while self.listening:
            try:
                # Intentar obtener un frame de la queue
                # queue.get(timeout) lanza queue.Empty si expira timeout
                frame = self.q.get(timeout=timeout or 1.0)
                yield frame
            except queue.Empty:
                # Timeout agotado, sin frames en queue
                # Si ya no estamos escuchando, terminar el generador
                if not self.listening:
                    break
                # Si aún estamos escuchando, reintentar (esperar más frames)
                continue
