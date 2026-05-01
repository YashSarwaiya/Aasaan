"""Hybrid RAG over structured forms via Haystack 2.x.

Why this exists: in v3 the eval's "RAG mode" used stock Qwen with the full
structured form stuffed into the prompt — naive, no actual retrieval. That
collapsed to 34% accuracy. Real RAG (chunk + embed + retrieve top-K) over
the per-field structured form is what we need.

Design:
  - Each schema field of each document becomes one chunk (e.g.
    "doc_42 / current_medication: Aspirin 81mg, Spiriva 10mcg").
  - Embed each chunk with a small sentence-transformer model.
  - At query time: embed the user question, retrieve top-K chunks across
    all docs (or filter by user_id in production), feed to the LLM.

Production multi-tenancy is handled via metadata filters: each chunk carries
{user_id, doc_id, field_name}. Same vector store, isolated reads.

Lazy imports — Haystack is a heavy optional dep. Falls back to a simple
local in-memory vector store using sentence-transformers directly if Haystack
isn't installed.

Usage:
    from pipeline.rag import build_rag_index, rag_answer

    rag = build_rag_index(structured_forms, lookup_fields=["current_medication", ...])
    answer = rag_answer(rag, question="What meds is patient on?", model=qwen, tok=tok)
"""

from __future__ import annotations

import json
from typing import Any


# Default fields to index for retrieval. Lookup fields are the natural targets
# for RAG (facts that the model shouldn't memorize). Reasoning fields can be
# retrieved too but are usually better answered by the trained model.
DEFAULT_INDEX_FIELDS_HINT = (
    "current_medication", "vitals", "laboratory_data",
    "physical_examination", "procedure_details",
    "preprocedure_diagnosis", "postprocedure_diagnosis",
    "patient_age", "patient_sex", "allergies",
)


def _flatten_field_value(value: Any) -> str:
    """Render a structured value (str / list / dict) into a single string for embedding."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(v).strip() for v in value if v)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items() if v)
    return str(value)


def _build_chunks(
    structured_items: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert structured forms → list of {text, meta} chunks.

    One chunk per (document, field). Empty values skipped.
    """
    chunks: list[dict[str, Any]] = []
    for item in structured_items:
        doc_id = item.get("id", item.get("doc_id", "unknown"))
        structured = item.get("structured", item)
        keys = fields if fields else list(structured.keys())
        for field in keys:
            if field not in structured:
                continue
            text = _flatten_field_value(structured[field])
            if not text:
                continue
            chunks.append({
                "text": f"{field}: {text}",
                "meta": {
                    "doc_id": str(doc_id),
                    "field": field,
                    "user_id": item.get("user_id", "default"),
                },
            })
    return chunks


def build_rag_index(
    structured_items: list[dict[str, Any]],
    *,
    fields: list[str] | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Build an in-memory vector index of structured-form chunks.

    Returns an opaque RAG handle. Use rag_retrieve() / rag_answer() with it.

    Tries Haystack first; falls back to a minimal sentence-transformers
    index if Haystack isn't available.
    """
    chunks = _build_chunks(structured_items, fields=fields)
    if not chunks:
        return {"_backend": "empty", "chunks": []}

    try:
        from haystack import Document
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.embedders import (
            SentenceTransformersDocumentEmbedder,
            SentenceTransformersTextEmbedder,
        )
        from haystack.components.retrievers.in_memory import (
            InMemoryEmbeddingRetriever,
        )

        store = InMemoryDocumentStore()
        docs = [Document(content=c["text"], meta=c["meta"]) for c in chunks]

        doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
        doc_embedder.warm_up()
        embedded = doc_embedder.run(documents=docs)["documents"]
        store.write_documents(embedded)

        text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)
        text_embedder.warm_up()
        retriever = InMemoryEmbeddingRetriever(document_store=store)

        return {
            "_backend": "haystack",
            "store": store,
            "text_embedder": text_embedder,
            "retriever": retriever,
            "n_chunks": len(chunks),
        }
    except ImportError:
        print(
            "  ⚠️  Haystack not installed; using minimal sentence-transformers fallback. "
            "Install: pip install haystack-ai sentence-transformers",
            flush=True,
        )

    # Fallback: minimal local index
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer(embedding_model)
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return {
            "_backend": "fallback",
            "model": model,
            "chunks": chunks,
            "embeddings": embeddings,
            "n_chunks": len(chunks),
        }
    except ImportError:
        print(
            "  ⚠️  No embedding library available. RAG disabled.",
            flush=True,
        )
        return {"_backend": "disabled", "chunks": chunks}


def rag_retrieve(rag, query: str, *, top_k: int = 5, user_id: str | None = None):
    """Retrieve top-K most relevant chunks for a query."""
    backend = rag.get("_backend")
    if backend == "disabled" or backend == "empty":
        return []

    if backend == "haystack":
        text_embedder = rag["text_embedder"]
        retriever = rag["retriever"]
        embedded = text_embedder.run(text=query)["embedding"]
        filters = (
            {"field": "meta.user_id", "operator": "==", "value": user_id}
            if user_id else None
        )
        results = retriever.run(
            query_embedding=embedded, top_k=top_k, filters=filters
        )
        return [
            {"text": d.content, "meta": d.meta, "score": d.score}
            for d in results["documents"]
        ]

    if backend == "fallback":
        import numpy as np
        model = rag["model"]
        chunks = rag["chunks"]
        embeddings = rag["embeddings"]
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = embeddings @ q_emb
        if user_id:
            mask = np.array([c["meta"].get("user_id") == user_id for c in chunks])
            scores = np.where(mask, scores, -np.inf)
        idxs = np.argsort(-scores)[:top_k]
        return [
            {"text": chunks[i]["text"], "meta": chunks[i]["meta"], "score": float(scores[i])}
            for i in idxs
            if scores[i] > -np.inf
        ]

    return []


def rag_answer(
    rag,
    question: str,
    model,
    tokenizer,
    *,
    top_k: int = 5,
    user_id: str | None = None,
    max_tokens: int = 200,
) -> str:
    """End-to-end RAG: retrieve top-K chunks, stuff into prompt, generate answer."""
    from .llm import ask

    chunks = rag_retrieve(rag, question, top_k=top_k, user_id=user_id)
    if not chunks:
        context = "(no relevant context found in knowledge base)"
    else:
        context = "\n".join(f"- {c['text']}" for c in chunks)

    prompt = f"""You are answering questions using ONLY the retrieved context below.

CONTEXT:
{context}

QUESTION: {question}

Answer concisely (1-3 sentences). If the context doesn't contain the answer, say "Not in the knowledge base." Do not invent facts.
Answer:"""
    return ask(model, tokenizer, prompt, max_tokens=max_tokens)
