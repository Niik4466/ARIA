# A.R.I.A Documentation

Welcome to the internal documentation for **A.R.I.A** (Advanced Responsive Intelligent Assistant).

A.R.I.A is designed with a deeply modular and highly robust node-based processing system, leaning on cutting-edge local AI technologies encompassing LangGraph logic routing, Local LLMs (Ollama), Retrieval-Augmented Context (ChromaDB), and Real-Time RVC Voice Synthesis.

## Navigation

This `docs` folder contains multiple targeted files that completely demystify the internal workings of the assistant:

- 🧠 **[Architecture & Graph Logic](architecture.md)**
  Deep dive into the `src/graph.py` file, examining how LangGraph passes context nodes, and how the LLM determines whether it should execute a tool, query the Knowledge Base, or respond directly.

- 🧩 **[Core Components Breakdown](components.md)**
  Examines exactly what files like `agent.py`, `rag.py`, `asr.py`, and `tts.py` do. Outlines the separation of responsibilities, including the specific `WakeWordSetup` wizard mechanics.

- ⚙️ **[Configuration Tutorial](configuration.md)**
  A quick-start guide referencing `config.py`. Explains how to manipulate A.R.I.A's logic thresholds, speed, voice models, RVC morphing, and memory retention behavior.

- 🎤 **[RVC Installation](rvc_installation.md)**
  Complete tutorial for installing RVC Voice Converter using RVC-WebUI

- 🧠 **[RAG Knowledge Base Tutorial](rag_tutorial.md)**
  Quick tutorial on how to use the rag system

- ⚡ **[Flash Attention](flash_attention.md)**
  Guide to configure and install Flash Attention for improved TTS inference times and memory usage (NVIDIA only).

## Source Code Organization (`src/`)

The `src` directory separates responsibilities strictly to allow massive modularity:
- **`src/graph/`**: Contains the LangGraph implementation (`edges.py`, `nodes.py`, `state.py`) dictating the conversational flow.
- **`src/Tools/`**: Hosts all functional sub-routines (OS, Math, API wrappers) executed dynamically when the agent determines it needs real-world data.
- **`src/vad/`**: Contains the speech isolation mechanisms and the WakeWord similarity algorithms (`vad.py`, `wakeword.py`).
- **`src/tts/`**: Groups text-to-speech mechanisms (`kokoro_tts.py`, `qwen3_tts.py`) and the `rvc_backend.py`.
- **`src/container.py`**: The Dependency Injection engine loading and mapping models into a unified scope.
- **`src/utils.py`**: Helper scripts, mainly the `Config` parsing logic.
- **`src/agent.py` & `src/rag.py` & `src/asr.py`**: Standalone bridge scripts querying LLMs, VectorDBs, and Speech Recognizers.