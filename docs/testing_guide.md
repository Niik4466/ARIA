# A.R.I.A Testing Guide

This document outlines the testing infrastructure for A.R.I.A, explaining what tests exist, why they are implemented, and how to execute them efficiently.

A.R.I.A relies heavily on deep learning models (Faster-Whisper, Kokoro TTS, Qwen-TTS, Silero VAD, WavLM, Ollama, and ChromaDB). Testing these components naively would cause severe delays and memory exhaustion. Our testing strategy isolates the system's logic from these expensive computations using extensive mocking, while also providing performance testing tools to measure authentic latency when needed.

## 1. Unit Testing Suite (`tests/unit/`)

### What it is
The unit testing suite validates the logic, data transformations, and state transitions of every module in the `src/` directory without invoking actual machine learning models or making real system calls (like audio playback).

### Why it exists
- **Regression Safety**: Ensures that changes to core flow (like LangGraph definitions, array processing, config reading, tool extraction) do not break the program.
- **Speed**: By aggressively mocking heavy dependencies via `pytest-mock` and `unittest.mock` within `tests/unit/conftest.py`, the entire test suite completes in a few seconds rather than minutes.
- **Component Isolation**: Isolates RAG logic, MCP manager fetching routing, Wakeword mathematical thresholds, state transitions, and config handlers.

### Included Test Modules
- `test_utils.py` & `test_container.py`: Structural DI tests and config parsing.
- `test_audio_io.py`: Multithreading queue assertions and resampling math logic.
- `test_asr.py` & `tts/test_tts.py`: Base speech model mapping verifications.
- `vad/test_vad.py`: Audio chunk threshold probabilities and WavLM cosine similarity logic.
- `test_agent.py`: Regex cleansing algorithms over thoughts `<think>`.
- `test_rag.py`: ChromaDB hierarchy and weighting formulas.
- `mcp/test_mcp_manager.py`: Tool mapping arrays extraction.
- `graph/test_nodes.py`: LangGraph conditional nodes transitions (Routing rules). 

### How to Execute Unit Tests
To run the full suite quickly, use the local Virtual Environment Python:

```bash
# Run all unit tests
venv/bin/python -m pytest tests/unit/ -v

# Run a specific unit test file
venv/bin/python -m pytest tests/unit/graph/test_nodes.py -v
```

---

## 2. End-to-End Performance Tests (`tests/test_graph_performance.py`)

### What it is
A specialized test script that measures the actual latency mapping of the LangGraph execution path using real model abstractions without blocking the environment for audio inputs.

### Why it exists
To evaluate bottlenecks across Node transitions. It measures key operations such as:
- **ASR Transcription Time**
- **Agent Generation/Tool Execution Time**
- **RAG Retrieval Latencies**
- **Time-to-First-Speech (TTFS)** and **Streaming Gap Delays**.

It consumes a "Golden Dataset" (`tests/golden_dataset.json`) categorized logically (e.g., `general`, `rag_search`, `tool_execution`) to force the graph into different distinct behavioral paths. The environment's audio input/output is muted and mocked to simulate streaming gap intervals accurately based on sample rates without producing local sounds.

### How to Execute Performance Tests
Running these tests **will process real prompts through your configured A.R.I.A setup** (Ollama models, TTS inference). Ensure you have adequate hardware memory free.

```bash
# Export the latency timings to a CSV results sheet
venv/bin/python -m pytest tests/test_graph_performance.py -s
```

After execution, a file named `performance_results.csv` will be generated inside the `tests/` folder. This file contains precise timing metrics organized by `category`, `prompt_id`, `operation`, `mean_time`, and `std_time`.

---

## 3. Native Model Execution Integration Tests (`tests/integration/`)

### What it is
The integration testing suite validates whether the hardcoded machine-learning model tags/strings embedded securely across A.R.I.A modules are valid HuggingFace/PyTorch Hub repositories.

### Why it exists
Unit tests mock out all AI model executions to prioritize speed. However, if a developer misspells a model ID string (like altering `all-MiniLM-L6-v2` to `all-MiniLM-L6-v3` natively), standard unit tests will ignorantly ignore this failure. Integration Tests catch this behavior by actually executing the local module initialization blocks on real physical threads to assert their loading behavior.

To prevent GPU Memory Leaks (`CUDA Out Of Memory`) that appear when attempting to load heavy models directly consecutively, these tests utilize isolated Python **Multiprocessing**. Each AI component is tested natively—chaining its real internal dependencies based strictly on definitions dictated by `config.json` instead of using injected mocks—inside a separate native subprocess with a maximum threshold of `10.0 seconds`. If the weights successfully start bootstrapping (no explicit crashing) before the clock terminates, the subprocess is gracefully wiped freeing up RAM and moving correctly to the next component.

### How to Execute Integration Tests
```bash
venv/bin/python -m pytest tests/integration/ -v
```
