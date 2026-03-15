# RAG System (Knowledge Base)

ARIA is capable of reading your local documents and utilizing that information strictly in her vocal responses. The workflow mapping is extremely simple:

1. By default, the system scans for documents within the `./documents` directory (this path can be altered in `config.py`).
2. **Category Organization**: Create subdirectories inside `./documents`. Each folder acts as an isolated knowledge "category".
    * Example: `./documents/Physics/` for Physics class notes.
    * Example: `./documents/Manuals/` for Technical device manuals.
3. Drop your **.pdf** or **.txt** files within these categories:
   *Pro Tip: Ensure the filenames are highly descriptive for optimal vector relevance matching.*
4. Upon boot-up, ARIA scans, vectorizes, chunk-separates, and permanently indexes these files to her logical vector database (`.chroma_db`).
5. Over the course of a conversation, if you inquire about a related subject, the local Agent will automatically traverse the category matching to pull relevant chunked information before processing the final textual or vocal output response.
