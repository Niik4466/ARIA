"""
Módulo de Speech-to-Text (STT) usando Faster Whisper.
Genera audio en memoria (como bytes WAV) sin escribir en disco.
"""

import io
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from typing import List
from config import FASTER_WHISPER_MODEL, FASTER_WHISPER_DEVICE, SAMPLE_RATE


class FasterWhisperSTT:
    """
    Transcriptor de voz usando Faster Whisper.
    
    Flujo:
    1. Recibe lista de frames numpy (int16) del micrófono
    2. Concatena los frames en un array de audio
    3. Genera bytes WAV en memoria (io.BytesIO)
    4. Pasa los bytes a Whisper para transcripción
    5. Retorna el texto transcrito
    """
    
    def __init__(self):
        """
        Carga el modelo de Whisper (se configura en config.py).
        
        SAMPLE_RATE: 16000 Hz (requerido por WebRTC VAD)
        FASTER_WHISPER_DEVICE: 'cpu' o 'cuda' (para GPU)
        FASTER_WHISPER_MODEL: 'tiny', 'base', 'small', 'medium', 'large-v3', etc.
        """
        self.model = WhisperModel(FASTER_WHISPER_MODEL, device=FASTER_WHISPER_DEVICE)
        self.sample_rate = SAMPLE_RATE

    def _frames_to_wav_bytes(self, frames: List[np.ndarray]) -> bytes:
        """
        Convierte frames de audio PCM int16 a bytes WAV en memoria.
        
        Args:
            frames: Lista de arrays numpy int16, cada uno representa un frame de audio.
                   Típicamente ~320 muestras @ 16kHz = 20ms de audio.
        
        Returns:
            bytes: Contenido WAV completo en memoria (formato WAV/PCM_16).
        
        Proceso:
        1. Concatenar todos los frames en un array único
        2. Asegurar que es int16
        3. Usar soundfile para codificar a WAV en un buffer de memoria (io.BytesIO)
        4. Retornar los bytes WAV
        """
        # Concatenar todos los frames en un único array de audio
        audio = np.concatenate(frames).astype('int16')
        
        # Crear buffer en memoria para el WAV
        wav_buffer = io.BytesIO()
        
        # Escribir audio al buffer como WAV (PCM 16-bit)
        sf.write(wav_buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        
        # Obtener los bytes WAV del buffer
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.getvalue()
        
        return wav_bytes

    def transcribe(self, frames: List[np.ndarray]) -> str:
        """
        Transcribe una lista de frames de audio a texto en español.
        
        Args:
            frames: Lista de arrays numpy int16 (frames PCM del micrófono).
        
        Returns:
            str: Texto transcrito (unión de segmentos con espacios).
        
        Proceso interno:
        1. Convertir frames a bytes WAV en memoria (_frames_to_wav_bytes)
        2. Pasar bytes WAV a Whisper (whisper.transcribe con format='wav')
        3. Extraer texto de cada segmento
        4. Unir segmentos con espacios
        5. Retornar texto final
        
        Notas:
        - Whisper puede leer WAV directamente desde bytes usando formato 'wav'
        - language='es' fuerza la detección de idioma español
        - beam_size=1 es más rápido (beam_size=5+ es más preciso pero lento)
        """
        # Generar WAV en memoria
        wav_bytes = self._frames_to_wav_bytes(frames)
        
        # Pasar bytes WAV a Whisper (usando io.BytesIO para simular archivo)
        wav_buffer = io.BytesIO(wav_bytes)
        
        # Transcribir con Whisper
        # segments: lista de segmentos con .text (el texto del segmento)
        # _info: metadatos (ignorados aquí)
        segments, _info = self.model.transcribe(
            wav_buffer,
            language="es",           # Forzar español
            beam_size=1              # Velocidad vs precisión
        )
        
        # Extraer texto de cada segmento y unir con espacios
        text = " ".join(seg.text.strip() for seg in segments).strip()
        
        return text
