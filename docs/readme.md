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