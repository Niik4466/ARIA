# ARIA (Adaptative, Responsive, Intelligent, Assistant)

![Python](https://img.shields.io/badge/Python-3.10%2B-4B8BBE?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Run-black?logo=ollama&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-5A32A3)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)
![Edge-TTS](https://img.shields.io/badge/TTS-Edge_TTS-0078D4)
![Kokoro](https://img.shields.io/badge/TTS-Kokoro-6C757D)
![Faster-Whisper](https://img.shields.io/badge/STT-Faster_Whisper-6C757D)
![RVC](https://img.shields.io/badge/RVC-Voice_Conversion-6C757D)


**ARIA** es un asistente virtual avanzado diseñado para ser adaptable e inteligente. Combina lo mejor de los LLMs locales (vía Ollama) con capacidades de conversión de voz (RVC) y un sistema de memoria basado en documentos (RAG), permitiendo una interacción natural y personalizada.

## ✨ Funcionalidades

-   **Respuestas por voz**: Utiliza motores TTS de alta calidad (Kokoro o Edge-TTS) mejorados con RVC (Retrieval-based Voice Conversion) para clonar voces con realismo.
-   **Sistema RAG (Retrieval-Augmented Generation)**: Capacidad para leer, indexar y recordar información de tus propios documentos PDF y archivos de texto.
-   **Uso de Herramientas**: ARIA puede decidir cuándo usar herramientas externas o buscar en su base de conocimientos para responder mejor a tus preguntas.
-   **Ejecución Local**: Todo el sistema (STT, LLM, TTS, RVC) puede ejecutarse localmente protegiendo tu privacidad.

---

## 📚 Sistema RAG (Base de Conocimiento)

ARIA puede leer tus documentos para usar esa información en sus respuestas. El flujo de trabajo es simple:

1.  Por defecto, el sistema busca documentos en el directorio `./documents` (o el configurado en `config.py`).
2.  **Organización por Categorías**: Crea subdirectorios dentro de `./documents`. Cada subdirectorio actúa como una "categoría" de conocimiento.
    *   Ejemplo: `./documents/Física/` para apuntes de física.
    *   Ejemplo: `./documents/Manuales/` para manuales técnicos.
3.  Coloca tus archivos **.pdf** o **.txt** dentro de esas carpetas.
4.  Al iniciar, ARIA escaneará, vectorizará y guardará estos documentos en su base de datos vectorial (`.chroma_db`).
5.  Cuando preguntes algo relacionado, el agente buscará la información más relevante en esa categoría.

---

## ⚙️ Configuración (`config.py`)

El archivo `config.py` es el centro de control de ARIA. Aquí puedes ajustar el comportamiento del agente.

### Audio & STT (Speech to Text)
*   **`FASTER_WHISPER_MODEL`**: Modelo de reconocimiento de voz (ej: "small", "medium", "large-v3").
*   **`FASTER_WHISPER_DEVICE`**: Dispositivo de inferencia ("cpu" o "cuda").

### Agente (LLM)
*   **`API_URL`**: Dirección de la API de Ollama.
*   **`RESPONSE_MODEL`**: Modelo principal para conversar (ej: "qwen3").
*   **`DECISOR_MODEL`**: Modelo ligero para tomar decisiones de enrutamiento (ej: "qwen3:0.6b").
*   **`AGENT_EXTRA_PROMPT`**: Prompt del sistema para definir la personalidad (ej: "Eres Miku...").

### TTS (Text to Speech) & RVC
*   **`USE_KOKORO`**: `True` para usar Kokoro (mejor calidad), `False` para Edge-TTS.
*   **`KOKORO_LANG`**: Idioma para Kokoro (ej: `'e'` para inglés/mix, ver doc oficial).
*   **`KOKORO_VOICE`**: Nombre de la voz base de Kokoro (ej: `'af_sarah'`).
*   **`KOKORO_SPEED`**: Velocidad del habla (ej: `0.94`).
*   **`VOICE_NAME`**: Voz de respaldo si se usa Edge-TTS (ej: `"es-CL-LorenzoNeural"`).
*   **`USE_RVC`**: `True` para aplicar conversión de voz RVC sobre el TTS.
*   **`RVC_MODEL`**: Nombre de la carpeta del modelo RVC en `./rvc/rvc_models/`.

### RAG
*   **`DOCUMENTS_PATH`**: Ruta donde se almacenan los documentos para la base de conocimiento.

---

## 🎙️ Configurar Voz Personalizada

Para que ARIA hable con una voz específica (ej. un personaje de anime o celebridad):

1.  **Descargar**: Busca y descarga un modelo RVC (normalmente un .zip) desde [voice-models.com](https://voice-models.com) o HuggingFace.
2.  **Instalar**: Descomprime el archivo y coloca la carpeta resultante dentro de `ARIA/rvc/rvc_models/`.
    *   La estructura debe ser: `./rvc/rvc_models/NOMBRE_DEL_MODELO/` (adentro deben estar el `.pth` y el `.index`).
3.  **Configurar ARIA**: Edita `config.py` y cambia `RVC_MODEL` por el nombre de esa carpeta.
    ```python
    RVC_MODEL = "NOMBRE_DEL_MODELO"
    ```
4.  **Ajustar Base**: Configura la voz base (Kokoro o Edge-TTS) para que se parezca lo más posible al tono deseado antes de la conversión:
    *   **Kokoro**: [Lista de voces disponibles](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
    *   **Edge-TTS**: [Lista de voces disponibles](https://tts.travisvn.com).
5.  **Velocidad**: Ajusta `KOKORO_SPEED` en `config.py` a tu gusto.

---

## 🚀 Instalación

### Prerequisitos
*   **Ollama**: Debe estar instalado y funcionando. [Descargar Ollama](https://ollama.com/download).

### Pasos
1.  Clona este repositorio o descarga el código.
2.  Ejecuta el script de instalación correspondiente a tu sistema operativo. Asegúrate de especificar si usas GPU (NVIDIA `cuda` o AMD `rocm`) para instalar las versiones correctas de PyTorch.

    **Windows:**
    ```bat
    install-windows.bat --gpu cuda
    ```
    *(O usa `install-windows.bat --help` para ver más opciones).*

    **Linux:**
    ```bash
    ./install-linux.sh --gpu cuda
    ```
    *(O usa `./install-linux.sh --help` para ver más opciones).*

    **MacOS:**
    ```bash
    ./install-mac.sh
    ```

3.  Una vez finalizada la instalación, puedes iniciar el sistema con los scripts de inicio automático (estos se encargarán de levantar Ollama, la API de RVC y la aplicación principal):

    *   **Windows**: Ejecuta `init.bat`.
    *   **Linux/Mac**: Ejecuta `./init.sh`.

------

> 💡 **Notas:**
> *   Los scripts de instalación para Linux y MacOS no han sido probados a fondo. Si encuentras algún problema, por favor, abre un `issue` en el repositorio.
> *   El sistema ha sido probado en una GPU NVIDIA RTX5070. Si experimentas un rendimiento lento en un sistema con menor capacidad, se recomienda reducir el tamaño (billones de parámetros) del modelo LLM utilizado en la configuración.
> *   Falta mejorar la herramienta de búsqueda por internet. Actualmente sirve para preguntar sobre noticias.
> *   Se recomienda que el nombre de los documentos almacenados por RAG sea descriptivo.
> *   Se recomienda el uso del LLM qwen3 en sus distintos sabores.
