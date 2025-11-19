# Arquitectura del Asistente de Voz

## Descripción General

Sistema de asistente de voz interactivo que:
- Escucha mientras se presiona **Ctrl+Alt+Espacio**
- Transcribe voz a texto (STT con Whisper)
- Genera respuestas con LLM local (Ollama)
- Reproduce respuesta en audio (TTS con edge-tts)
- **Todo en RAM, sin archivos temporales**

## Flujo de Datos (End-to-End)

```
USUARIO presiona Ctrl+Alt+Espacio
        ↓
listen_for_hotkey() detecta (on_press event)
        ↓
mic.start_listening() → callback comienza a encolar frames
        ↓
process_audio() thread obtiene frames vía mic.frames()
        ↓
VAD (UtteranceSegmenter) segmenta frames en utterances
        ↓
Para cada utterance: build_graph().invoke(state)
        │
        ├─→ STT Node
        │   - Frames (List[np.ndarray]) → WAV en memoria
        │   - Whisper transcribe a texto en español
        │   - Output: state["user_text"]
        │
        ├─→ LLM Node (Agent)
        │   - user_text → llamada HTTP a Ollama
        │   - Ollama genera respuesta (puede incluir <think> tags)
        │   - Output: state["reply_text"]
        │
        └─→ TTS Node
            - reply_text → filtro de <think> tags
            - edge-tts genera audio en memoria
            - sounddevice reproduce
            - Output: Audio reproducido + reply_text limpio
        ↓
USUARIO suelta Ctrl+Alt+Espacio
        ↓
listen_for_hotkey() detecta (on_release event)
        ↓
mic.stop_listening() → callback detiene de encolar frames
        ↓
Último utterance se procesa
        ↓
Vuelve a estado de espera
```

## Módulos

### 1. `main.py` - Orquestador Principal

**Responsabilidad**: Coordinar threads y puntos de entrada/salida del usuario.

**Threads**:
- `listen_for_hotkey()`: Global hotkey listener (pynput)
  - Detecta **Ctrl+Alt+Espacio** presionado → `mic.start_listening()`
  - Detecta **Ctrl+Alt+Espacio** soltado → `mic.stop_listening()`
  - Detecta **Ctrl+C** → `stop_event.set()` (cierre global)

- `process_audio()`: Procesador de audio
  - Espera a que `mic.listening=True`
  - Obtiene frames de `mic.frames()`
  - Invoca VAD para segmentación
  - Para cada utterance: `app.invoke(state)`

**Estado global**:
- `stop_event`: Event para señalizar cierre entre threads

### 2. `src/audio_io.py` - Captura de Micrófono

**Responsabilidad**: Capturar audio PCM en tiempo real organizando en frames.

**Clase**: `MicrophoneStream`
- **Flujo**: sounddevice → callback `_callback()` → queue → iterator `frames()`
- **Control**: `start_listening()` / `stop_listening()` para activar/desactivar encolamiento
- **Thread-safe**: Lock para sincronizar acceso a `listening` flag
- **Frame size**: FRAME_MS (20ms) = 320 muestras @ 16kHz
- **Formato**: PCM int16 mono

**Métodos clave**:
```python
mic.start_listening()      # Activar: callback encola frames
mic.frames()               # Iterator: yield frames si listening=True
mic.stop_listening()       # Desactivar: callback no encola, vaciar queue
```

### 3. `src/vad.py` - Segmentador de Utterances

**Responsabilidad**: Agrupar frames en "utterances" (frases completas) usando VAD.

**Clase**: `UtteranceSegmenter`
- **Algoritmo**: Máquina de estados (speaking=False/True)
- **Entrada**: Iterator de frames numpy int16
- **Salida**: Iterator de listas de frames (cada lista = utterance)

**VAD Logic** (WebRTC VAD):
1. Frame speech → añadir a buffer
2. Buffer lleno (>= min_speech_ms) → marcar speaking=True
3. Silencio consecutivo (> max_silence_ms) → emitir utterance, reset
4. Fin de stream → emitir último utterance

**Parámetros**:
```python
segmenter = UtteranceSegmenter(
    min_speech_ms=300,      # Mínimo para considerar habla válida
    max_silence_ms=600      # Máximo de silencio dentro de utterance
)
```

### 4. `src/stt.py` - Transcripción (Speech-to-Text)

**Responsabilidad**: Convertir frames de audio a texto en español.

**Clase**: `FasterWhisperSTT`
- **Modelo**: Faster Whisper (configurable: tiny/base/small/medium/large-v3)
- **Entrada**: List[np.ndarray] (frames PCM)
- **Proceso**:
  1. Concatenar frames en un array
  2. **Generar WAV en memoria** (io.BytesIO, sin disco)
  3. Pasar bytes WAV a Whisper
  4. Whisper transcribe a texto español
  5. Retornar texto
- **Salida**: str (texto transcrito)

**Key Method**:
```python
text = stt.transcribe(utterance_frames)  # → List[frames] to str
```

### 5. `src/agent.py` - LLM (Language Model)

**Responsabilidad**: Generar respuestas conversacionales.

**Función**: `call_ollama(prompt: str) → str`
- **Conexión**: HTTP POST a Ollama local (típicamente http://127.0.0.1:11434/api/generate)
- **Entrada**: Texto user_text
- **Proceso**:
  1. Construir prompt: `{SYSTEM_PROMPT}\nUsuario: {user_text}\nAsistente:`
  2. Enviar a Ollama vía REST API
  3. Ollama genera respuesta (puede incluir `<think>...</think>` tags si el modelo lo soporta)
  4. Retornar respuesta
- **Salida**: str (respuesta del LLM)

**Requisito**: Ollama corriendo localmente

```bash
ollama serve                    # Terminal 1: servir Ollama
ollama pull qwen3              # Terminal 2: descargar modelo
```

### 6. `src/graph.py` - Grafo de Procesamiento (LangGraph)

**Responsabilidad**: Orquestar el pipeline STT → LLM → TTS.

**Estado**: `GraphState` (TypedDict)
```python
{
    "frames": List[np.ndarray],   # Entrada: frames de micrófono
    "user_text": str,              # Intermedio: STT output
    "reply_text": str              # Salida: respuesta final (limpia)
}
```

**Nodos**:
1. **stt_node**: frames → user_text
2. **agent_node**: user_text → reply_text (con posibles `<think>` tags)
3. **tts_node**: reply_text → audio reproducido
   - **Filtrado**: Elimina `<think>...</think>` y tags sueltos
   - Utiliza regex para limpieza
   - Almacena reply_text limpio en estado

**Uso**:
```python
app = build_graph()
result = app.invoke({"frames": utterance_frames})
# result["user_text"] = lo que dijo el usuario
# result["reply_text"] = respuesta limpia
# Audio ya fue reproducido por tts_node
```

### 7. `src/tts.py` - Conversión de Texto a Audio

**Responsabilidad**: Convertir texto a audio y reproducirlo.

**Clase**: `TTSEngine`
- **Provider**: edge-tts (Microsoft Azure TTS)
- **Entrada**: str (texto en español)
- **Proceso**:
  1. `edge_tts.Communicate.stream()` genera chunks de audio
  2. **Acumular en bytearray** (memoria, no disco)
  3. Decodificar bytes (soundfile o ffmpeg fallback)
  4. Reproducir con sounddevice
  5. Esperar a que termine
- **Salida**: Audio reproducido (efecto secundario)

**Configuración**:
```python
VOICE_NAME = "es-ES-AlvaroNeural"  # Español de España, voz masculina
```

**Key Method**:
```python
tts.speak(text)  # str → audio reproducido
```

## Flujo de Control de Teclas

### Ctrl+Alt+Espacio (Hold to Record)

```
Usuario presiona Ctrl y mantiene:
  pressed_keys = {Key.ctrl_l, ...}

Usuario presiona Alt y mantiene (3 teclas ahora):
  pressed_keys = {Key.ctrl_l, Key.alt_l, ...}

Usuario presiona Espacio (combinación completa):
  has_ctrl=True, has_alt=True, has_space=True
  → on_press() → mic.start_listening()
  → frames se encolan
  → VAD segmenta
  → Procesar utterances

Usuario suelta una tecla (ej. Espacio):
  → on_release() detecta combo quebrada
  → mic.stop_listening()
  → frames dejan de encolarse
  → Último utterance se procesa
```

### Ctrl+C (Exit)

```
Usuario presiona Ctrl y mantiene
Usuario presiona C
  → on_press() detecta Ctrl+C
  → stop_event.set()
  → listener.join() retorna False
  → Cierre limpio de threads
```

## Características de Diseño

### 1. Sin Archivos Temporales
- ✅ Audio capturado: frames en memoria (numpy arrays)
- ✅ STT: WAV generado en io.BytesIO (memoria), no en disco
- ✅ TTS: Audio generado en bytearray, reproducido directamente
- ✅ Ningún archivo escrito a tmp_audio/ o similar

### 2. Escucha bajo Demanda (Press & Hold)
- ✅ Listener global (no bloquea entrada estándar)
- ✅ Escucha mientras se mantiene presionado
- ✅ Procesamiento paralelo en otro thread
- ✅ Terminación automática al soltar

### 3. Filtrado de Tags de Pensamiento
- ✅ STT/LLM pueden producir `<think>...</think>` tags
- ✅ Nodo TTS los filtra antes de reproducir
- ✅ Regex para eliminar bloques completos y tags sueltos
- ✅ reply_text guardado limpio en estado

### 4. Thread-Safe
- ✅ MicrophoneStream usa locks para `listening` flag
- ✅ Queue thread-safe para frames
- ✅ stop_event sincroniza cierre entre threads
- ✅ No hay race conditions críticas

## Configuración (config.py)

```python
# Audio
SAMPLE_RATE = 16000         # Hz (requerido: 8000/16000/32000/48000)
CHANNELS = 1                # Mono
FRAME_MS = 20               # ms (10/20/30, requerido por VAD)
VAD_AGGRESSIVENESS = 2      # 0-3 (balance ruido/sensibilidad)

# STT
FASTER_WHISPER_MODEL = "small"   # Modelo Whisper
FASTER_WHISPER_DEVICE = "cpu"    # o "cuda"

# LLM
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen3"      # Modelo local

# TTS
VOICE_NAME = "es-ES-AlvaroNeural"  # Voz edge-tts
```

## Requisitos y Dependencias

### Paquetes Python (requirements.txt)
```
sounddevice==0.4.6          # Reproducción/captura de audio
webrtcvad==2.0.10           # Voice Activity Detection
numpy==1.26.4               # Arrays numéricos
faster-whisper==1.0.3       # STT
langgraph==0.2.49           # Grafo de procesamiento
pydantic==2.9.2             # Validación
requests==2.32.3            # HTTP (Ollama)
edge-tts==6.1.14            # TTS
soundfile==0.12.1           # Lectura/escritura de audio
pynput==1.7.6               # Global hotkey listener
```

### Requisitos Externos
- **Ollama**: Servicio local para LLM
  - Instalación: https://ollama.ai
  - Comando: `ollama serve`
  - Modelo: `ollama pull qwen3`

- **FFmpeg** (opcional pero recomendado):
  - Para decodificar audio en memoria
  - Windows: `choco install ffmpeg` o descargar de ffmpeg.org

## Casos de Uso

### 1. Conversación Normal
```
Usuario: "¿Cuál es la capital de España?"
STT: "¿Cuál es la capital de España?" 
LLM: "La capital de España es Madrid."
TTS: Reproduce "La capital de España es Madrid."
```

### 2. Modelo con Pensamiento
```
Usuario: "¿Cuánto es 234 * 567?"
LLM: "<think>234 * 567 = ... = 132678</think>La respuesta es 132678."
Filtrado: "La respuesta es 132678."
TTS: Reproduce solo "La respuesta es 132678."
```

### 3. Error de Conexión
```
Usuario presiona Ctrl+Alt+Espacio pero Ollama no corre
Error: ConnectionError en call_ollama()
Stack trace mostrado en terminal
Esperar a que Ollama se reinicie
```

## Troubleshooting

| Problema | Causa | Solución |
|----------|-------|----------|
| No se detecta habla | VAD muy agresivo | Bajar VAD_AGGRESSIVENESS en config.py |
| Muchos falsos positivos | VAD muy sensible | Subir VAD_AGGRESSIVENESS |
| STT lento | Modelo muy grande | Cambiar a "tiny" o "base" en config.py |
| Ollama no responde | No corriendo | `ollama serve` en otra terminal |
| Audio distorsionado | Micrófono roto o volumen alto | Revisar nivel de volumen del micrófono |
| TTS falla | ffmpeg no instalado, MP3 no soportado | Instalar ffmpeg, o usar wav |

## Notas de Desarrollo

### Extendibilidad

1. **Cambiar STT**: Reemplazar `src/stt.py` con otra librería (Google Speech-to-Text, AssemblyAI, etc.)
2. **Cambiar LLM**: Cambiar endpoint Ollama o usar OpenAI API
3. **Cambiar TTS**: Reemplazar `src/tts.py` (Google TTS, Elevenlabs, etc.)
4. **Agregar componentes**: Extender `src/graph.py` con nuevos nodos

### Performance

- **STT**: ~2-5s para 10s de audio (depende del modelo)
- **LLM**: 1-10s por respuesta (depende del modelo y complejidad)
- **TTS**: ~500ms por respuesta (depends de longitud)
- **Latencia total**: ~5-20s por turno conversacional

### Debugging

Habilitar logging en módulos (agregar al inicio de archivos):
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

---

**Última actualización**: Noviembre 2025  
**Estado**: Totalmente documentado, sin archivos temporales, hotkey press-and-hold funcional
