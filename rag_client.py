"""
RAG client: discovers ChromaDB collections, retrieves the most relevant
NASA document chunks for a question and formats them as LLM context.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from llm_client import resolve_base_url

# Longest excerpt (in characters) passed to the LLM for a single chunk.
MAX_CHUNK_CHARS = 1500

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories (any folder whose name contains "chroma")
    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower() and not d.name.startswith(".")
    ]

    for chroma_dir in sorted(chroma_dirs):
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collections = client.list_collections()

            for collection in collections:
                # chromadb >= 0.6 returns Collection objects; older versions return names
                collection_name = getattr(collection, "name", collection)
                key = f"{chroma_dir.name}::{collection_name}"
                try:
                    count = client.get_collection(collection_name).count()
                except Exception:
                    count = "unknown"
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection_name,
                    "display_name": f"{collection_name} ({chroma_dir.name}, {count} chunks)",
                    "document_count": str(count),
                }

        except Exception as e:
            error_text = str(e)
            backends[f"{chroma_dir.name}::error"] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (unavailable: {error_text[:60]}"
                                f"{'...' if len(error_text) > 60 else ''})",
                "document_count": "0",
            }

    return backends


def _query_embedding_function(collection_metadata: Optional[Dict]) -> Optional[OpenAIEmbeddingFunction]:
    """Build the OpenAI embedding function used to embed user questions.

    The question must be embedded with the same model that embedded the
    documents, which the pipeline records in the collection metadata.
    """
    api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = (collection_metadata or {}).get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    kwargs = {"api_key": api_key, "model_name": model}
    base_url = resolve_base_url(api_key)
    if base_url:
        kwargs["api_base"] = base_url
    return OpenAIEmbeddingFunction(**kwargs)


class RetrievalCollection:
    """Thin wrapper that embeds the question with OpenAI and queries ChromaDB."""

    def __init__(self, collection, embedding_function):
        self.collection = collection
        self.embedding_function = embedding_function
        self.name = collection.name

    def count(self) -> int:
        return self.collection.count()

    def query(self, query_text: str, n_results: int, where: Optional[Dict] = None) -> Dict:
        if self.embedding_function is None:
            raise RuntimeError("No OpenAI API key available to embed the question.")
        query_embedding = self.embedding_function([query_text])[0]
        return self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)

    Returns (collection, success, error_message) as expected by chat.py.
    """
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(collection_name)
        wrapped = RetrievalCollection(collection, _query_embedding_function(collection.metadata))
        return wrapped, True, None
    except Exception as e:
        return None, False, str(e)


def _deduplicate_and_sort(results: Dict) -> Dict:
    """Remove repeated chunks and order the hits by similarity (best first)."""
    if not results or not results.get("documents") or not results["documents"][0]:
        return results

    docs = results["documents"][0]
    metas = (results.get("metadatas") or [[{}] * len(docs)])[0]
    dists = (results.get("distances") or [[0.0] * len(docs)])[0]
    ids = (results.get("ids") or [[""] * len(docs)])[0]

    best: Dict[str, tuple] = {}
    for doc, meta, dist, doc_id in zip(docs, metas, dists, ids):
        key = " ".join((doc or "").split()).lower()
        if key and (key not in best or dist < best[key][2]):
            best[key] = (doc, meta, dist, doc_id)

    ordered = sorted(best.values(), key=lambda item: item[2])
    return {
        "ids": [[item[3] for item in ordered]],
        "documents": [[item[0] for item in ordered]],
        "metadatas": [[item[1] for item in ordered]],
        "distances": [[item[2] for item in ordered]],
    }


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    if collection is None or not query or not query.strip():
        return None

    where_filter = None
    if mission_filter and mission_filter.lower() not in ("all", "all missions", "none", ""):
        where_filter = {"mission": mission_filter}

    # Ask for a few extra hits so that k unique chunks remain after de-duplication.
    raw = collection.query(query.strip(), n_results=n_results + 2, where=where_filter)
    cleaned = _deduplicate_and_sort(raw)

    if cleaned and cleaned.get("documents"):
        for field in ("ids", "documents", "metadatas", "distances"):
            if cleaned.get(field):
                cleaned[field] = [cleaned[field][0][:n_results]]
    return cleaned


def _pretty(value: str) -> str:
    return str(value).replace("_", " ").title()


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["NASA mission archive excerpts (most relevant first):", ""]

    seen = set()
    source_number = 0
    for doc, metadata in zip(documents, metadatas or [{}] * len(documents)):
        key = " ".join((doc or "").split()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        source_number += 1
        metadata = metadata or {}

        mission = _pretty(metadata.get("mission", "unknown mission"))
        source = metadata.get("source", "unknown source")
        category = _pretty(metadata.get("document_category", "general document"))

        context_parts.append(
            f"[Source {source_number}] Mission: {mission} | Document: {source} | Category: {category}"
        )

        text = doc.strip()
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + " ..."
        context_parts.append(text)
        context_parts.append("-" * 60)

    return "\n".join(context_parts)
