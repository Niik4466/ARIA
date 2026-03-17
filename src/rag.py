"""
Módulo RAG (Retrieval-Augmented Generation).

Propósito:
- Escanear /documents/ para encontrar subdirectorios (categorías).
- Cargar PDFs y archivos de texto.
- Generar embeddings y almacenarlos en ChromaDB.
- Proveer funciones de búsqueda por categoría.
- Generar descripciones de categorías usando LLM.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import requests
from .utils import Config
config = Config()

RESPONSE_MODEL = config.get("RESPONSE_MODEL")
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
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            url=API_URL.replace("/api/generate", ""),
            model_name="nomic-embed-text"
        )
        self.history_collection = self.client.get_or_create_collection(
            name="conversation_history",
            embedding_function=self.embedding_fn
        )

        
    def _read_pdf(self, path: str) -> str:
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Limpieza del texto extraído
            text = " ".join(text.split()) # Reemplaza múltiples espacios y saltos de línea con un solo espacio
            text = text.strip() # Elimina espacios al inicio y final
            return text
        except Exception as e:
            print(f"[RAG] Error leyendo PDF {path}: {e}")
            return ""

    def _read_text(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[RAG] Error leyendo texto {path}: {e}")
            return ""
            
    def _generate_description(self, category: str, file_names: list) -> str:
        """Genera una breve descripción de la categoría usando el LLM."""
        prompt = (
            f"Analiza la categoría '{category}' que contiene los siguientes archivos: {', '.join(file_names)}.\n"
            "Genera una descripción MUY BREVE (máximo 15 palabras) de qué tipo de información contiene esta categoría.\n"
            "Ejemplo: 'Documentación técnica sobre python y scripts.'\n"
            "Descripción:"
        )
        
        try:
            payload = {
                "model": RESPONSE_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            }
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                desc = res.json().get("response", "").strip()
                # Limpiar si el modelo es verboso
                desc = desc.replace('"', '').replace("'", "")
                return desc
        except Exception as e:
            print(f"[RAG] Error generando descripción: {e}")
        
        return f"Información sobre {category}."

    def _should_update_file(self, file_path: str, filename: str, collection) -> bool:
        """Verifica si un archivo necesita ser actualizado basándose en mtime."""
        current_mtime = os.path.getmtime(file_path)
        existing = collection.get(
            where={"source": filename},
            limit=1,
            include=["metadatas"]
        )
        
        if existing['ids']:
            stored_mtime = existing['metadatas'][0].get('last_modified', 0)
            if current_mtime == stored_mtime:
                return False
            else:
                print(f"[RAG] {filename} ha cambiado. Re-indexando...")
                collection.delete(where={"source": filename})
                return True
        return True

    def _update_file(self, file_path: str, filename: str, collection, current_mtime: float):
        """Lee, fragmenta y almacena un archivo en la colección."""
        content = ""
        if filename.lower().endswith(".pdf"):
            content = self._read_pdf(file_path)
        elif filename.lower().endswith((".txt", ".md")):
            content = self._read_text(file_path)
        
        if not content:
            return

        chunk_size = 1000 
        overlap = 200
        
        chunks = []
        for i in range(0, len(content), chunk_size - overlap):
            chunks.append(content[i:i + chunk_size])
        
        chunks = [c for c in chunks if len(c) > 50]

        if not chunks:
            return

        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        print(f"[RAG] Vectorizando {filename} ({len(chunks)} chunks)...")
        
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_meta = [{
                "source": filename, 
                "last_modified": current_mtime
            } for _ in batch_chunks]
            
            collection.add(
                documents=batch_chunks,
                metadatas=batch_meta,
                ids=batch_ids
            )

    def _process_category_files(self, cat_path: str, collection) -> tuple[list, bool]:
        """Procesa todos los archivos de una categoría. Retorna lista de archivos y si hubo cambios."""
        files_processed = []
        category_changed = False
        
        for filename in os.listdir(cat_path):
            file_path = os.path.join(cat_path, filename)
            if not os.path.isfile(file_path): 
                continue
            
            if self._should_update_file(file_path, filename, collection):
                category_changed = True
                self._update_file(file_path, filename, collection, os.path.getmtime(file_path))
                
            files_processed.append(filename)
            
        return files_processed, category_changed

    def _handle_category_description(self, category: str, collection, files_processed: list, category_changed: bool) -> str:
        """Gestiona la descripción de la categoría (generación o recuperación)."""
        current_collection_meta = collection.metadata or {}
        stored_description = current_collection_meta.get("description")
        final_description = ""

        if category_changed or not stored_description:
            if files_processed:
                print(f"[RAG] Generando nueva descripción para {category}...")
                final_description = self._generate_description(category, files_processed)
                collection.modify(metadata={"description": final_description})
            else:
                final_description = "Categoría vacía."
        else:
            final_description = stored_description
            
        return final_description

    def update(self) -> dict:
        """
        Escanea DOCUMENTS_DIR, actualiza vectores SOLO si hay cambios y retorna metadata.
        """
        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)
            print(f"[RAG] Directorio {DOCUMENTS_DIR} creado.")
            return {}

        categories_metadata = {}
        
        for category in os.listdir(DOCUMENTS_DIR):
            cat_path = os.path.join(DOCUMENTS_DIR, category)
            if not os.path.isdir(cat_path):
                continue
                
            print(f"[RAG] Verificando categoría: {category}...")
            
            try:
                collection = self.client.get_or_create_collection(
                    name=category,
                    embedding_function=self.embedding_fn
                )
            except Exception as e:
                print(f"[RAG] Error accediendo a colección {category}: {e}")
                continue

            files_processed, category_changed = self._process_category_files(cat_path, collection)
            description = self._handle_category_description(category, collection, files_processed, category_changed)
            categories_metadata[category] = description
            
        print("[RAG] Sincronización completa.")
        return categories_metadata

    def query_category(self, category: str, query_text: str, n_results: int = 3) -> str:
        """Busca en una categoría específica."""
        try:
            collection = self.client.get_collection(name=category, embedding_function=self.embedding_fn)
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Formatear contexto
            context = ""
            for i, doc_list in enumerate(results['documents']):
                for j, doc in enumerate(doc_list):
                    source = results['metadatas'][i][j].get('source', 'unknown')
                    context += f"--- Fragmento de {source} ---\n{doc}\n\n"
            
            return context
        except Exception as e:
            return f"[RAG Error] No se pudo buscar en la categoría {category}: {e}"

    def add_to_history(self, user_text: str, agent_response: str):
        """Añade un extracto de conversación al historial RAG."""
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
        """Busca en el historial de conversación los extractos más relevantes."""
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

# Instancia global
rag_manager = RAGManager()
