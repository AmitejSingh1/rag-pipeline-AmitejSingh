# RAG Pipeline – Internship Assignment

A minimal, modular Retrieval-Augmented Generation (RAG) pipeline that ingests `.txt`/`.pdf` files, chunks them, creates embeddings, indexes with FAISS, retrieves relevant context, and generates grounded answers. Includes a CLI and a Streamlit UI.

## Features
- Ingest `.txt` and `.pdf`
- Word-based chunking with overlap
- Embeddings via `sentence-transformers`
- Vector search via FAISS (inner product)
- Generation via OpenAI (if `OPENAI_API_KEY` set) or multiple free local models:
  - **Encoder-Decoder**: Flan-T5 (small/base)
  - **Decoder-Only**: Gemma (2B/7B), Mistral (7B), Llama-2 (7B/13B)
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

**Note for Llama/Mistral models**: Some models require HuggingFace authentication. If you get an error, run:
```powershell
pip install huggingface-hub
huggingface-cli login
```
Then accept the model license on https://huggingface.co when prompted.

**System Requirements**: Larger models (7B+) need 8GB+ RAM. Start with smaller models (flan-t5-small, gemma-2b) if you have limited resources.

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

## Deployment

### Streamlit Cloud (Recommended)

1. **Push your code to GitHub** (already done: `rag-pipeline-AmitejSingh`)

2. **Go to Streamlit Cloud**: https://share.streamlit.io/

3. **Sign in with GitHub** and click "New app"

4. **Configure your app**:
   - **Repository**: Select `AmitejSingh1/rag-pipeline-AmitejSingh`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`

5. **Add secrets** (optional, for API keys):
   - Click "Advanced settings" → "Secrets"
   - Add secrets:
     ```
     OPENAI_API_KEY=sk-your-key-here
     HUGGINGFACE_HUB_TOKEN=hf-your-token-here
     ```
   - **Also add** (for Python 3.13 compatibility):
     ```
     PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
     ```

6. **Deploy**: Click "Deploy" and wait ~2 minutes

7. **Your app will be live at**: `https://rag-pipeline-amitej-singh.streamlit.app`

**Note**: On Streamlit Cloud, the `indexes/` directory is ephemeral (resets on restart). Users will need to rebuild the index after uploading documents.

### Docker (optional)
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


