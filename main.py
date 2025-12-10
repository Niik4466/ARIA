"""
Asistente de Voz Interactivo - Punto de Entrada.

Arquitectura General:
┌─────────────────────────────────────────────────────────────┐
│                    MAIN                                     │
│  Orquesta dos threads paralelos:                            │
│  1. listen_for_hotkey: Detecta Ctrl+Alt+Espacio (presionar) │
│  2. process_audio: Procesa audio grabado en tiempo real     │
└─────────────────────────────────────────────────────────────┘
        │
        ├─→ MicrophoneStream: Captura audio del micrófono
        │   (start_listening/stop_listening para control manual)
        │
        ├─→ UtteranceSegmenter: VAD para segmentar en frases
        │
        └─→ build_graph(): Pipeline STT→LLM→TTS

Flujo de Ejecución (cuando usuario presiona Ctrl+Alt+Espacio):

1. listen_for_hotkey() detecta Ctrl+Alt+Espacio
2. mic.start_listening() → frames se encolan en MicrophoneStream.q
3. Mientras se mantiene presionado:
   - process_audio() obtiene frames de mic.frames()
   - UtteranceSegmenter.segment() agrupa frames en utterances (via VAD)
   - Para cada utterance: build_graph().invoke() ejecuta STT→LLM→TTS
4. Usuario suelta Ctrl+Alt+Espacio
5. mic.stop_listening() → frames dejan de encolarse
6. Último utterance se procesa
7. Vuelve a estado de espera

Teclas:
- Ctrl+Alt+Espacio (mantener presionado): Grabar y procesar
- Ctrl+C: Salir del programa (exit)
"""

import threading
import sys
import queue
from src.audio_io import MicrophoneStream
from src.graph import build_graph


# Evento global para señalizar cierre del programa
stop_event = threading.Event()


def listen_for_hotkey(mic: MicrophoneStream, audio_queue: "queue.Queue"):
    """
    Thread-local hotkey listener: Detecta Ctrl+Alt+Espacio y Ctrl+C.
    
    Uso de pynput.keyboard.Listener:
    - Listener es global (no bloqueado por entrada estándar)
    - on_press: se dispara cuando se presiona una tecla
    - on_release: se dispara cuando se suelta una tecla
    - Retornar False detiene el listener
    
    Estados del sistema:
    - is_recording=False → en espera
    - is_recording=True → grabando (mic.listening=True)
    
    Cambios de estado:
    - on_press: Ctrl+Alt+Espacio → start_listening (is_recording=True)
    - on_release: suelta cualquier tecla de la combo → stop_listening (is_recording=False)
    - on_press: Ctrl+C → stop_event.set() (cierre global)
    
    Importante:
    - pressed_keys es un set que se actualiza en on_press/on_release
    - Permite detectar combinaciones (Ctrl AND Alt AND Espacio)
    - Se limpia después de cada uso para no generar falsos positivos
    """
    from pynput import keyboard

    pressed_keys = set()
    is_recording = False

    def on_press(key):
        nonlocal is_recording
        
        try:
            # Añadir tecla al set de teclas presionadas
            pressed_keys.add(key)

            # ════════════════════════════════════════════════════
            # Detector de Ctrl+C (salida del programa)
            # ════════════════════════════════════════════════════
            has_ctrl = keyboard.Key.ctrl_l in pressed_keys or keyboard.Key.ctrl_r in pressed_keys
            if has_ctrl and isinstance(key, keyboard.KeyCode) and key.char == 'c':
                print("\n[Main]👋 Cerrando por Ctrl+C…")
                stop_event.set()  # Señal global de cierre
                return False  # Detiene el listener de pynput

            # ════════════════════════════════════════════════════
            # Detector de Ctrl+Alt+Espacio (inicio de grabación)
            # ════════════════════════════════════════════════════
            has_ctrl = keyboard.Key.ctrl_l in pressed_keys or keyboard.Key.ctrl_r in pressed_keys
            has_alt = keyboard.Key.alt_l in pressed_keys or keyboard.Key.alt_r in pressed_keys
            has_space = keyboard.Key.space in pressed_keys

            if has_ctrl and has_alt and has_space and not is_recording:
                print("[Main] 🎤 Ctrl+Alt+Espacio presionado → Escuchando…")
                is_recording = True
                mic.start_listening()

        except (AttributeError, KeyError):
            # Teclas especiales o excepciones de pynput
            pass

    def on_release(key):
        nonlocal is_recording
        
        try:
            # Remover tecla del set
            pressed_keys.discard(key)

            # ════════════════════════════════════════════════════
            # Detector de final de Ctrl+Alt+Espacio
            # ════════════════════════════════════════════════════
            # Si se suelta CUALQUIERA de las 3 teclas, detenemos grabación
            has_ctrl = keyboard.Key.ctrl_l in pressed_keys or keyboard.Key.ctrl_r in pressed_keys
            has_alt = keyboard.Key.alt_l in pressed_keys or keyboard.Key.alt_r in pressed_keys
            has_space = keyboard.Key.space in pressed_keys

            # Si estamos grabando pero la combo se rompió → parar
            if is_recording and not (has_ctrl and has_alt and has_space):
                print("[Main] 🔴 Teclas soltadas → Deteniendo grabación…")
                mic.stop_listening()
                is_recording = False

                frames = mic.drain_frames()
                if frames:
                    audio_queue.put(frames)

        except (AttributeError, KeyError):
            pass

    # Iniciar listener de pynput (bloqueante)
    # Se ejecuta hasta que algún callback retorne False
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def process_audio(mic: MicrophoneStream, app, audio_queue: "queue.Queue"):
    """
    Thread paralelo: Consume bloques de frames (uno por pulsación) desde
    `audio_queue` y los procesa como un único prompt (STT→LLM→TTS).
    """
    while not stop_event.is_set():
        try:
            frames = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Procesar todas las frames capturadas durante la pulsación como UN SOLO prompt
        print("[Main] 🗣️  Procesando audio capturado…")
        state = {"frames": frames}
        result = app.invoke(state)
        print(f"[Main] 👤 Tú: {result.get('user_text', '')}")
        print(f"[Main] 🤖 Bot: {result.get('reply_text', '')}")
        print("\n")


def main():
    """
    Punto de entrada principal.
    
    Inicializa:
    1. MicrophoneStream: Captura de audio en tiempo real
    2. UtteranceSegmenter: Segmentador VAD
    3. build_graph(): Pipeline STT→LLM→TTS
    4. Dos threads paralelos:
       - listen_for_hotkey: Detector global de Ctrl+Alt+Espacio y Ctrl+C
       - process_audio: Procesador de audio bajo demanda
    
    Uso:
    - python main.py
    - Ctrl+Alt+Espacio (mantener) para grabar y procesar
    - Ctrl+C para salir
    
    Control de flujo:
    - stop_event.is_set() chequea si el usuario presionó Ctrl+C
    - Cuando se dispara, ambos threads salen de sus loops
    """
    print("🎙️  Asistente de voz a la espera.")
    print("   Ctrl+Alt+Espacio (mantener presionado) → Escuchar y grabar")
    print("   Ctrl+C → Salir")

    # Construir componentes
    app = build_graph()

    # Mantener micrófono abierto durante toda la sesión
    with MicrophoneStream() as mic:
        # Cola para pasar bloques de frames (capturados durante cada pulsación)
        audio_queue = queue.Queue()

        # Thread 1: Detector de hotkeys
        hotkey_thread = threading.Thread(
            target=listen_for_hotkey,
            args=(mic, audio_queue),
            daemon=True  # Muere si main() termina
        )

        # Thread 2: Procesador de audio
        process_thread = threading.Thread(
            target=process_audio,
            args=(mic, app, audio_queue),
            daemon=True  # Muere si main() termina
        )

        # Iniciar ambos threads
        hotkey_thread.start()
        process_thread.start()

        try:
            # Main espera a que stop_event se dispare (Ctrl+C)
            while not stop_event.is_set():
                threading.Event().wait(0.2)
        except KeyboardInterrupt:
            # Fallback por si acaso 
            print("\n[Main]👋 Cerrando…")
            stop_event.set()


if __name__ == "__main__":
    main()
