import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # Existing ingestion and retrieval settings...
    "corpus_dir": "data/corpus",
    "raw_docs_dir": "data/raw/docs",
    "chroma_dir": "chroma_db",
    "collection_name": "fastapi_docs",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 256,
    "chunk_overlap": 60,
    "max_embedding_tokens": 240,
    "similarity_metric": "cosine",
    "similarity_threshold": 0.35,
    "retrieval_mode": "similarity",  # "similarity" or "mmr"
    "top_k": 12,
    "fetch_k": 20,
    "mmr_lambda": 0.6,

    # Provider configuration: loaded exclusively from .env
    "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),

    # Replace with the exact sentence required by Lesson 2
    "refusal_sentence": "I cannot answer this question based on the provided documentation.",
}
