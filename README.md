# RAG Pipeline – Internship Assignment

A minimal, modular Retrieval-Augmented Generation (RAG) pipeline that ingests `.txt`/`.pdf` files, chunks them, creates embeddings, indexes with FAISS, retrieves relevant context, and generates grounded answers. Includes a CLI and a Streamlit UI.

## Features
- Ingest `.txt` and `.pdf`
- Word-based chunking with overlap
- Embeddings via `sentence-transformers`
- Vector search via FAISS (inner product)
- Generation via OpenAI (if `OPENAI_API_KEY` set) or local `flan-t5-small`
- CLI and Streamlit frontend
- Persisted index and metadata to disk

## Project Structure
```
.
├─ docs/
│  └─ ARCHITECTURE.md        # Design decisions
├─ data/                      # Place your .txt/.pdf here
├─ indexes/                   # Saved FAISS index and metadata
├─ src/rag/
│  ├─ config.py
│  ├─ ingest.py
│  ├─ chunk.py
│  ├─ embeddings.py
│  ├─ index_faiss.py
│  ├─ retriever.py
│  ├─ generator.py
│  └─ pipeline.py
├─ app.py                     # CLI entry
├─ streamlit_app.py           # UI entry
├─ requirements.txt
└─ README.md
```

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional: Set OpenAI (for higher-quality generation)
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

## Usage (CLI)
1) Build index from documents in `data/`
```powershell
python app.py build --docs_dir data --index_dir indexes
```

2) Query the index
```powershell
python app.py query --index_dir indexes --question "What is RAG?"
```

## Usage (Streamlit UI)
```powershell
streamlit run streamlit_app.py
```

## Design Notes
- Retrieval uses cosine similarity via inner product over normalized vectors.
- Generation falls back to local `flan-t5-small` when `OPENAI_API_KEY` is absent.
- Index persistence stores FAISS index and JSON metadata for reproducibility.

## Docker (optional)
Build and run:
```powershell
docker build -t rag-pipeline .
docker run -it --rm -p 8501:8501 -v ${PWD}/data:/app/data rag-pipeline
```

## Architecture & Design Decisions
See `docs/ARCHITECTURE.md`.

## Tests (placeholder)
Add your tests under `tests/` using `pytest`.

## License
MIT


