# DEPRECADO MOMENTANEAMENTE
# FLUJO MANEJADO CON BOTONES, A FUTURO CONFIGURABLE CON VAD


"""
Módulo de Voice Activity Detection (VAD) usando WebRTC VAD.

Propósito: Segmentar un stream de audio frames en 'utterances' (frases completas)
detectando automáticamente dónde comienza y termina el habla.

Flujo de datos:
1. Recibe iterator de frames numpy int16
2. Cada frame se pasa a WebRTC VAD para detectar si es habla o silencio
3. Acumula frames de habla en buffer
4. Cuando detecta suficiente silencio, emite un utterance (lista de frames)
5. Reinicia el buffer y espera siguiente utterance

Concepto clave:
- VAD (Voice Activity Detection): algoritmo que distingue voz de ruido/silencio
- Utterance: secuencia de frames que representa una frase hablada
- Este módulo agrupa frames en utterances para pasarlos a STT
"""

import webrtcvad
import numpy as np
from typing import Iterator, List
from config import SAMPLE_RATE, FRAME_MS, VAD_AGGRESSIVENESS


def _to_bytes(frame: np.ndarray) -> bytes:
    """
    Convierte un frame numpy int16 a bytes para WebRTC VAD.
    
    WebRTC VAD requiere input como bytes, no como arrays numpy.
    
    Parámetro:
    - frame: Array numpy int16 (típicamente 320 muestras)
    
    Retorna: Bytes (representación binaria del frame)
    """
    # frame.tobytes() convierte array int16 a bytes (16 bits por muestra)
    return frame.tobytes()


class UtteranceSegmenter:
    """
    Segmentador de audio que agrupa frames en utterances usando WebRTC VAD.
    
    Uso:
        segmenter = UtteranceSegmenter(min_speech_ms=300, max_silence_ms=600)
        for utterance_frames in segmenter.segment(mic.frames()):
            process_utterance(utterance_frames)
    
    Atributos configurables:
    - min_speech_ms: Duración mínima de habla para considerar un utterance válido
    - max_silence_ms: Duración máxima de silencio dentro de un utterance
      (si hay más silencio, se cierra el utterance)
    
    Lógica interna:
    1. VAD clasifica cada frame como speech o non-speech (silencio/ruido)
    2. Acumula frames de habla en voiced_buffer
    3. Cuenta frames de silencio consecutivos
    4. Cuando se llena el buffer mínimo, marcamos speaking=True
    5. Cuando hay demasiado silencio, emitimos el utterance
    """
    
    def __init__(self, min_speech_ms=300, max_silence_ms=600):
        """
        Inicializa el segmentador.
        
        Parámetros:
        - min_speech_ms: Milisegundos mínimos de habla antes de considerar "hablando"
                        (evita falsos positivos con clicks o ruido breve)
                        Valor típico: 300ms = ~10-15 frames @ 20ms/frame
        
        - max_silence_ms: Milisegundos máximos de silencio permitidos dentro de un utterance
                         Si silencio > esto, cierra el utterance
                         Valor típico: 600ms = ~30 frames @ 20ms/frame
        
        VAD_AGGRESSIVENESS (de config.py):
        - 0: Menos agresivo (detecta más voz, menos falsos negativos)
        - 3: Más agresivo (solo sonidos fuertes = voz, menos ruido)
        - Valor típico: 2 (balance)
        """
        # Inicializar WebRTC VAD
        # Requiere SAMPLE_RATE de 8000, 16000, 32000, o 48000 Hz
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        
        # Convertir milisegundos a número de frames
        # FRAME_MS = 20ms (de config.py), so 300ms = 15 frames
        self.frames_per_window = max(1, min_speech_ms // FRAME_MS)
        
        # Número máximo de frames de silencio consecutivos permitidos
        # 600ms silencio = 30 frames @ 20ms/frame
        self.max_silence_frames = max(1, max_silence_ms // FRAME_MS)

    def segment(self, frames: Iterator[np.ndarray]) -> Iterator[List[np.ndarray]]:
        """
        Generador que produce utterances a partir de un stream de frames.
        
        Parámetro:
        - frames: Iterator de arrays numpy int16 (típicamente de MicrophoneStream.frames())
        
        Rendimiento: Yields listas de frames, donde cada lista es un utterance completo.
        
        Algoritmo (máquina de estados):
        
        Estado: speaking=False (esperando habla)
        - Si frame es speech: añadir a voiced_buffer, contar frames de habla
        - Si tenemos >= frames_per_window frames de habla: transicionar a speaking=True
        - Si frame es silencio: descartar (aún no estamos hablando, evita falsos positivos)
        
        Estado: speaking=True (dentro de un utterance)
        - Si frame es speech: añadir a voiced_buffer, resetear silence_count
        - Si frame es silencio: incrementar silence_count
        - Si silence_count >= max_silence_frames: emitir utterance, resetear
        
        Transiciones:
        - speaking=False → speaking=True: cuando se acumula suficiente habla
        - speaking=True → speaking=False: cuando hay demasiado silencio
        
        Al final: si termina con frames acumulados (ej. el usuario suelta Ctrl+Alt+Espacio),
        emitir ese utterance también.
        """
        # Buffer que acumula frames de un utterance en progreso
        voiced_buffer: List[np.ndarray] = []
        
        # Contador de frames de silencio consecutivos
        silence_count = 0
        
        # Bandera: ¿estamos dentro de un utterance?
        speaking = False

        # Procesar cada frame que llega del iterator
        for frame in frames:
            # Clasificar este frame: ¿es habla o no?
            # is_speech retorna True/False
            is_speech = self.vad.is_speech(_to_bytes(frame), SAMPLE_RATE)

            if is_speech:
                # ════════════════════════════════════════════
                # FRAME DE HABLA DETECTADO
                # ════════════════════════════════════════════
                
                # Añadir frame a buffer
                voiced_buffer.append(frame)
                
                # Resetear contador de silencio (ese fue speech, no silencio)
                silence_count = 0
                
                # ¿Aún no hemos transicionado a speaking=True?
                if not speaking and len(voiced_buffer) >= self.frames_per_window:
                    # Tenemos suficientes frames de habla, marcar como speaking
                    speaking = True
                    # (continuamos acumulando frames)
            else:
                # ════════════════════════════════════════════
                # FRAME DE SILENCIO/NO-HABLA DETECTADO
                # ════════════════════════════════════════════
                
                if speaking:
                    # Ya estamos dentro de un utterance, contar este silencio
                    silence_count += 1
                    
                    # ¿Hemos visto demasiado silencio? → Fin del utterance
                    if silence_count >= self.max_silence_frames:
                        # Emitir el utterance
                        yield voiced_buffer
                        
                        # Resetear para siguiente utterance
                        voiced_buffer = []
                        silence_count = 0
                        speaking = False
                else:
                    # Aún no estamos hablando, este frame silencioso no importa
                    # Vaciar buffer pequeño para no acumular ruido
                    voiced_buffer = []

        # ════════════════════════════════════════════════════════════════════
        # FIN DEL STREAM (cuando stop_listening() se llama)
        # ════════════════════════════════════════════════════════════════════
        # Si el stream termina con frames acumulados en voiced_buffer,
        # emitir ese utterance final también (ej. usuario suelta Ctrl+Alt+Espacio)
        if voiced_buffer:
            yield voiced_buffer
