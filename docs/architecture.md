# A.R.I.A - Architecture & Graph Logic

A.R.I.A operates on a state-based processing graph powered by **LangGraph**. This architecture allows the assistant to make intelligent decisions on an ongoing conversation, effectively routing the processing pipeline through a sequence of nodes based on AI-driven categorizations.

## The Core Graph (`src/graph.py`)

The main logic in `graph.py` orchestrates the dialogue loop by passing a shared `GraphState` dictionary between the different nodes. 

### Graph State Definitions
The `GraphState` keeps track of the current state of a single conversation loop:
- **`user_text`**: The initially transcribed audio from the user.
- **`history_context`**: Conversation history.
- **`tools_context`**: Text accumulated from executing various tools.
- **`rag_category` & `rag_context`**: The selected knowledge-base category and retrieved content.
- **`next_node` & `selected_category`**: Routing flags and decisions made by the `decisor` nodes.

### Node Pipeline

A normal interaction flows sequentially through these nodes, diverging at the Tool Decisor:

#### 1. `asr_node` 
- **Purpose**: Acts as the entry point of the pipeline.
- **Functionality**: Once the system detects the activation word, it starts listening to the user's speech via the ASR (Automatic Speech Recognition) component. It returns the transcribed `user_text`.

#### 2. `rag_decisor_node`
- **Purpose**: Determines whether external knowledge is needed.
- **Functionality**: Uses the `user_text` and a lightweight LLM decision prompt to figure out if the user is asking about a topic contained in the RAG (Retrieval-Augmented Generation) document collection. If relevant, it queries the `rag_manager` and injects the gathered context into `rag_context`.

#### 3. `tool_decisor_node`
- **Purpose**: Decides the next course of action (Tool execution or Direct Response).
- **Functionality**: Evaluates the `user_text`, `history_context`, and the newly acquired `rag_context`. Based on this information, the LLM selects a specific category. Valid categories include:
  - `search`: Web search.
  - `os`: Operating system interactions.
  - `basic`: Calculations and system time.
  - `autoconfig`: Re-configuring the assistant (e.g., wakeword).
  - `exit`: Request to shutdown.
  - `response`: Information is sufficient, proceed to generate the final response.

#### 4. `tool_node`
- **Purpose**: Generates and executes the tool JSON.
- **Functionality**: Once directed here, it generates a JSON payload for the chosen tool. It executes the tool from the `registry.py` and records the output in `tools_context`. Finally, it loops **back** to `tool_decisor_node` to determine if more tools are needed or if it's time to answer the user.
- *Wait Mechanism*: If processing begins here, A.R.I.A emits a quick "working on it" audio cue using TTS to prevent dead silence while executing tasks.

#### 5. `integrated_response_node`
- **Purpose**: Concludes the specific interaction.
- **Functionality**: Synthesizes a final natural language response utilizing the `user_text`, `rag_context`, and `tools_context`. It streams the response from the LLM directly into the TTS engine, playing audio asynchronously for minimal latency. It also logs the finalized Q&A into the RAG history context.

### The Main Loop (`run_aria`)
Outside the LangGraph execution, the system rests in a `while True` loop waiting for the **WakeWord** to trigger. When triggered, it plays an acknowledgment tone/phrase, initializes an empty `GraphState`, invokes the graph pipeline, and when finished (or interrupted), goes back to waiting for the WakeWord.
