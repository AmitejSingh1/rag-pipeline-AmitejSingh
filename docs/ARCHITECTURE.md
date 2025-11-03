# Architecture & Design Decisions

## Overview
This project implements a minimal Retrieval-Augmented Generation (RAG) system with a clean, modular architecture: ingest → chunk → embed → index → retrieve → generate. Each stage is pluggable.

## Components
- Ingestion (`src/rag/ingest.py`)
  - Loads `.txt` and `.pdf` (via `pypdf`). Outputs documents: `{id, source_path, text}`.
- Chunking (`src/rag/chunk.py`)
  - Word-based windows (default 300 words, 60 overlap) to balance recall and redundancy. Outputs `{doc_id, chunk_id, text, source_path}`.
- Embeddings (`src/rag/embeddings.py`)
  - `sentence-transformers/all-MiniLM-L6-v2` for speed and reasonable quality. Embeddings are L2-normalized `float32` for cosine similarity.
- Indexing (`src/rag/index_faiss.py`)
  - FAISS `IndexFlatIP` (inner product). Cosine similarity is achieved by normalizing vectors. Saves binary index and JSON metadata for reproducibility.
- Retrieval (`src/rag/pipeline.py`)
  - Encodes query, searches FAISS, returns top‑K passages with scores.
- Generation (`src/rag/generator.py`)
  - Prompt constructed to enforce grounding and inline citations `[doc#chunk]`.
  - Uses OpenAI Chat Completions if `OPENAI_API_KEY` present; otherwise falls back to local `flan-t5-small` via `transformers`.
- Orchestration (`src/rag/pipeline.py`)
  - `build_index(docs_dir)`, `load_index()`, `answer(question)`.

## Key Decisions
- Cosine similarity via `IndexFlatIP` + L2 normalization
  - Simpler and efficient baseline; avoids maintaining separate cosine implementations.
- Word-based chunking
  - Tokenizers vary across models; word counts are fast, transparent, and adequate for baseline. Overlap at 20% mitigates boundary loss.
- Persistence
  - Store FAISS index and metadata (`metadata.json`) in `indexes/` so retrieval is reproducible and startup is fast.
- Dual-generation paths
  - OpenAI for quality if available; local `flan-t5-small` ensures offline demo ability.

## Extensibility
- Stronger Embeddings: swap to `bge-base`, `e5-base`, or OpenAI embeddings.
- Reranking: add a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) after FAISS retrieval.
- Hybrid Search: fuse BM25 (Elastic/Whoosh) with dense scores.
- Advanced Chunking: token-aware (tiktoken), header/semantic boundaries, PDF structure-aware splitting.
- Scale: FAISS IVF/HNSW or a vector DB (Chroma/Weaviate/PGVector). Batch encoding.
- Guardrails: score thresholds, abstain/clarify prompts, quote-only answers.

## Limitations
- Local `flan-t5-small` has limited reasoning; intended as a fallback.
- No reranker in baseline; can impact precision on mixed corpora.
- PDF text relies on embedded text layer; scanned PDFs need OCR (e.g., Tesseract).

## Data Flow
1. `build_index`: ingest → chunk → embed → index → save artifacts.
2. `answer`: load index → embed query → search → build prompt → generate answer.

## Testing
- `tests/test_chunk.py`: sanity checks for chunking behavior.
- `tests/test_embeddings.py`: shape checks for embedding outputs.

## Security & Privacy
- No document uploads beyond local filesystem by default.
- If using OpenAI, only prompt text is sent; remove sensitive content or use the local generator.


