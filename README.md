# ARIA (Adaptive, Responsive, Intelligent Assistant)

![Python](https://img.shields.io/badge/Python-3.10%2B-4B8BBE?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Run-black?logo=ollama&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-5A32A3)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)

**A.R.I.A** is an advanced virtual assistant designed to be adaptable and intelligent. It orchestrates fast local LLM inference (via Ollama) with robust speech recognition (STT), highly customizable realistic audio synthesis (TTS + RVC), and dynamic memory retention (RAG).

Its main objective is to answer straightforward questions, execute tasks using system tools, and accompany the user in a fluid vocal manner with minimal latency.

## ✨ A.R.I.A's Capabilities

- **Exclusively Vocal Interaction**: Endlessly listens for its "wakeword" loop and responds dynamically via Voice Generation. Hands-Free entirely.
- **Dynamic Auto-Configuration**: A.R.I.A uses her synthesized voice to guide you step-by-step through configuring or changing the wakeword samples without requiring you to touch the code.
- **RAG Local Memory**: Drop PDF and Text documents into a folder, and A.R.I.A will parse, read, and strictly base her answers on your local documents.
- **Tool Automation**: If you ask for the time, weather, web search, or file management, her intelligent node Decisor will seamlessly interact with your PC instead of just chatting.
- **Humanized Voice Profiling**: Through optional RVC cloning working tightly with Kokoro TTS, the assistant sounds like any character you dictate.
- **100% Local & Private**: The STT (Faster-Whisper), brain (LLMs), and TTS all operate entirely on your own local GPU/CPU hardware. No cloud recording uploads.

---

## 📚 Technical Documentation

To thoroughly understand A.R.I.A's internals and configure her brain limits, review the manuals inside our `docs/` folder:

1. [⚙️ Quick Configuration Guide (`config.py`)](docs/configuration.md)
2. [🤖 Internal Architecture & Graph Logic nodes](docs/architecture.md)
3. [🧩 Component Purpose Breakdown (Agent, ASR, TTS, Wakeword)](docs/components.md)
4. [🧠 RAG Knowledge Base Tutorial](docs/rag.md)
5. [🎙️ Custom RVC Voices Installation](docs/rvc_installation.md)

---

## 🚀 Installation & Deployment

### Prequisites
* **Ollama**: Must be installed and running. [Download Ollama](https://ollama.com/download).

### Steps
1. Clone this repository or download the source code.
2. Run the correct installation script dependent on your Operating System. Make sure to specify your GPU stack (NVIDIA `cuda` or AMD `rocm`) for proper PyTorch wheels.

   **Windows:**
   ```bat
   install-windows.bat --gpu cuda
   ```

   **Linux:**
   ```bash
   ./install-linux.sh --gpu cuda
   ```

   **MacOS:**
   ```bash
   ./install-mac.sh
   ```

3. When finished, spin up all services (Ollama, RVC API, and Python Graph) with a single launch script:
   * **Windows**: `init.bat`
   * **Linux/Mac**: `./init.sh`

> 💡 **Note**: The system is heavily tested using Qwen3.5 (4b) on an RTX5070 achieving excellent sparse latency. If you possess heavier hardware constraints, scale down your LLMs within the configuration file to maintain speed.
