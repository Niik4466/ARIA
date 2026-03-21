import os
from collections import defaultdict
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .utils import Config

config = Config()

API_URL = config.get("API_URL")
DOCUMENTS_PATH = config.get("DOCUMENTS_PATH")
verbose_mode = config.get("verbose_mode")

_builtins_print = print
def print(*args, **kwargs):
    if verbose_mode:
        _builtins_print(*args, **kwargs)

RAG_DB_PATH = "./.chroma_db"
DOCUMENTS_DIR = DOCUMENTS_PATH

class RAGManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=RAG_DB_PATH)
        # Model instance
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )
        
        # Collection for hierarchical RAG documents
        self.docs_collection = self.client.get_or_create_collection(
            name="hierarchical_rag_docs",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Collection for conversation history
        self.history_collection = self.client.get_or_create_collection(
            name="conversation_history",
            embedding_function=self.embedding_fn
        )
        
        # Collection for MCP tools
        self.tools_collection = self.client.get_or_create_collection(
            name="mcp_tools",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )


    def _read_pdf(self, path: str) -> str:
        """Reads and extracts text from a PDF file."""
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            # Clean extracted text
            text = " ".join(text.split())
            text = text.strip()
            return text
        except Exception as e:
            print(f"[RAG] Error reading PDF {path}: {e}")
            return ""


    def _read_text(self, path: str) -> str:
        """Reads text from a standard text or markdown file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[RAG] Error reading text {path}: {e}")
            return ""


    def _should_update_file(self, file_path: str, filename: str) -> bool:
        """Checks if a file needs to be updated based on its modification time."""
        current_mtime = os.path.getmtime(file_path)
        existing = self.docs_collection.get(
            where={"source": filename},
            limit=1,
            include=["metadatas"]
        )
        
        if existing['ids'] and existing['metadatas']:
            stored_mtime = existing['metadatas'][0].get('last_modified', 0)
            if current_mtime == stored_mtime:
                return False
            else:
                print(f"[RAG] {filename} changed. Re-indexing...")
                self.docs_collection.delete(where={"source": filename})
                return True
        return True


    def _update_file(self, file_path: str, filename: str, category: str, current_mtime: float):
        """Reads, chunks, and stores a file in the documents collection."""
        content = ""
        if filename.lower().endswith(".pdf"):
            content = self._read_pdf(file_path)
        elif filename.lower().endswith((".txt", ".md")):
            content = self._read_text(file_path)
        
        if not content:
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_text(content)
        
        chunks = [c for c in chunks if len(c) > 50]

        if not chunks:
            return

        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        print(f"[RAG] Vectorizing {filename} ({len(chunks)} chunks)...")
        
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_meta = [{
                "source": filename, 
                "category": category,
                "last_modified": current_mtime
            } for _ in batch_chunks]
            
            try:
                self.docs_collection.add(
                    documents=batch_chunks,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
            except Exception as e:
                print(f"[RAG] Error adding batch for {filename}: {e}")


    def update(self) -> dict:
        """
        Scans DOCUMENTS_DIR, updates vectors ONLY if there are changes.
        Removes vectors for files or categories that no longer exist in the directory.
        Returns a dictionary of available categories mapping to generic descriptions.
        """
        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)
            print(f"[RAG] Directory {DOCUMENTS_DIR} created.")
            return {}

        categories_metadata = {}
        
        # Track valid files to clean up deleted ones later
        valid_files_on_disk = set()
        
        for category in os.listdir(DOCUMENTS_DIR):
            cat_path = os.path.join(DOCUMENTS_DIR, category)
            if not os.path.isdir(cat_path):
                continue
                
            print(f"[RAG] Checking category: {category}...")
            
            for filename in os.listdir(cat_path):
                file_path = os.path.join(cat_path, filename)
                if not os.path.isfile(file_path): 
                    continue
                
                valid_files_on_disk.add((category, filename))
                
                if self._should_update_file(file_path, filename):
                    self._update_file(file_path, filename, category, os.path.getmtime(file_path))
                    
            # Provide generic description instead of using LLM
            categories_metadata[category] = f"Information about {category}."
            
        # Clean up deleted files from ChromaDB
        existing_docs = self.docs_collection.get(include=["metadatas"])
        if existing_docs and existing_docs.get("metadatas"):
            sources_to_delete = set()
            for meta in existing_docs["metadatas"]:
                if not meta:
                    continue
                db_category = meta.get("category")
                db_source = meta.get("source")
                if (db_category, db_source) not in valid_files_on_disk:
                    sources_to_delete.add(db_source)
                    
            for source in sources_to_delete:
                print(f"[RAG] Removed missing file or category from index: {source}")
                self.docs_collection.delete(where={"source": source})

        print("[RAG] Sync complete.")
        return categories_metadata


    def retrieve_global(self, query: str, k: int = 20) -> list:
        """
        Hierarchical Level 1: Global Retrieval (Router).
        Retrieves the top-k documents globally across all categories.
        """
        try:
            results = self.docs_collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[RAG] Global retrieval error: {e}")
            return []
            
        retrieved_docs = []
        if results.get('documents') and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for doc, meta, dist in zip(docs, metas, distances):
                # Cosine space distance: similarity = 1 - distance
                similarity = 1.0 - dist
                category = meta.get("category", "unknown")
                source = meta.get("source", "unknown")
                retrieved_docs.append({
                    "text": doc,
                    "category": category,
                    "source": source,
                    "similarity": similarity
                })
        return retrieved_docs


    def weighted_category_vote(self, retrieved_docs: list) -> str:
        """
        Similarity-weighted voting to determine the most relevant category.
        """
        category_scores = defaultdict(float)
        for doc in retrieved_docs:
            category_scores[doc["category"]] += doc["similarity"]
        
        if not category_scores:
            return "none"
            
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        return best_category


    def query_category(self, category: str, query_text: str, n_results: int = 8) -> str:
        """
        Hierarchical Level 2: Restricted search within a specific category.
        Used internally by query_documents, but exposed if direct search is needed.
        """
        try:
            results = self.docs_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"category": category},
                include=["documents", "metadatas"]
            )
            
            context = ""
            if results.get('documents') and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    source = results['metadatas'][0][i].get('source', 'unknown')
                    context += f"--- Excerpt from {source} ---\n{doc}\n\n"
            
            return context.strip()
        except Exception as e:
            return f"[RAG Error] Could not search in category {category}: {e}"


    def query_documents(self, query: str) -> str:
        """
        Executes the full two-level hierarchical RAG retrieval.
        Returns the formatted context retrieved from the best category.
        """
        print(f"[📚 RAG] Performing hierarchical search for query...")
        
        # Level 1: Global retrieval & vote
        global_docs = self.retrieve_global(query, k=20)
        if not global_docs:
            print("[📚 RAG] No documents retrieved.")
            return ""
            
        best_category = self.weighted_category_vote(global_docs)
        print(f"[📚 RAG] Selected Category via voting: '{best_category}'")
        
        if best_category == "none":
            return ""
            
        # Level 2: Category retrieval
        return self.query_category(best_category, query, n_results=8)


    def add_to_history(self, user_text: str, agent_response: str):
        """Adds a conversation excerpt to the RAG history collection."""
        text = f"user: {user_text}\n agent: {agent_response}"
        import time
        idx = str(int(time.time() * 1000))
        try:
            self.history_collection.add(
                documents=[text],
                ids=[f"hist_{idx}"]
            )
        except Exception as e:
            print(f"[RAG] Error adding to history: {e}")


    def query_history(self, query_text: str, n_results: int = 3) -> str:
        """Searches the conversation history for the most relevant excerpts."""
        try:
            if self.history_collection.count() == 0:
                return ""
            
            n = min(n_results, self.history_collection.count())
            results = self.history_collection.query(
                query_texts=[query_text],
                n_results=n
            )
            
            context = ""
            if results and 'documents' in results and results['documents']:
                for doc in results['documents'][0]:
                    context += f"{doc}\n\n"
            
            return context.strip()
        except Exception as e:
            print(f"[RAG Error] Error querying history: {e}")
            return ""

    def write_tools(self, tools: list) -> None:
        """Stores or updates MCP tool descriptions in the tools collection."""
        if not tools:
            return
            
        ids = [t["id"] for t in tools]
        documents = [t["text"] for t in tools]
        metadatas = [t.get("metadata", {}) for t in tools]
        
        try:
            self.tools_collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"[RAG Error] Error writing tools: {e}")

    def get_tools(self, query: str, k: int = 3) -> list:
        """Retrieves semantically similar tools from the tools collection."""
        try:
            if self.tools_collection.count() == 0:
                return []
                
            n = min(k, self.tools_collection.count())
            results = self.tools_collection.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "metadatas"]
            )
            
            retrieved_tools = []
            if results.get('ids') and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for tool_id, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0]):
                    retrieved_tools.append({
                        "id": tool_id,
                        "text": doc,
                        "metadata": meta
                    })
            return retrieved_tools
        except Exception as e:
            print(f"[RAG Error] Error querying tools: {e}")
            return []
