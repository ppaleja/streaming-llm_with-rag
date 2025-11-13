# Retrieval-Augmented StreamingLLM (RAG) — Extension Notes

This document is a short developer-facing summary of the RAG skeleton added to
`streaming_llm/rag/`.

Purpose
- Index evicted segments from the streaming KV cache.
- Provide a retriever to fetch relevant past context when the model needs it.
- Offer reintegration helpers to feed retrieved passages back into the model.

Files added (skeleton)
- `streaming_llm/rag/store.py`: append-only store for evicted segments.
- `streaming_llm/rag/indexer.py`: incremental indexer wrapper (vector store).
- `streaming_llm/rag/retriever.py`: high-level retriever API.
- `streaming_llm/rag/trigger.py`: heuristics to decide when to retrieve.
- `streaming_llm/rag/integration.py`: conversion/reinsertion helper stubs.

Usage
- Register an eviction callback on the `StartRecentKVCache` implementation.
- Callback should add segments to `EvictedStore` and `Indexer`.
- During generation, call `Retriever.retrieve(query)` when `RetrievalTrigger`
  indicates retrieval is useful, then call `reintegrate_passages`.

Next steps
- Implement persistence and indexing (FAISS, LlamaIndex, or other).
- Implement reinsertion strategies (prompt prepend, fake past_key_values).
- Add benchmarks and integration tests.
