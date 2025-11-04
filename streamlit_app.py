import os
import json
import streamlit as st

from src.rag.config import RagConfig
from src.rag.pipeline import RagPipeline


st.set_page_config(page_title="RAG Demo", page_icon="📚", layout="wide")
st.title("RAG Pipeline Demo")

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more PDF or TXT files to build the index"
    )
    
    # Save uploaded files
    upload_dir = "data/uploads"
    use_uploaded = False
    
    if uploaded_files:
        os.makedirs(upload_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")
        use_uploaded = True
        with st.expander("📄 View uploaded files"):
            for f in uploaded_files:
                st.write(f"• {f.name} ({f.size / 1024:.1f} KB)")
    
    st.divider()
    st.header("Settings")
    
    # Show which source will be used
    if use_uploaded and uploaded_files:
        st.info("📤 Will use uploaded files to build index")
    else:
        st.info("📁 Will use documents from directory")
    
    docs_dir = st.text_input("Documents directory (if not using uploads)", value="data", disabled=use_uploaded and bool(uploaded_files))
    index_dir = st.text_input("Index directory", value="indexes")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, step=1)
    build_clicked = st.button("Build/Refresh Index")
    st.divider()
    backend = st.radio(
        "Generator backend",
        options=["Local (free)", "OpenAI (if key)"],
        index=0,
        help="Local uses free open-source models via transformers; OpenAI requires OPENAI_API_KEY.",
    )
    local_model = st.selectbox(
        "Local model",
        options=[
            # Encoder-decoder models (no auth required)
            "google/flan-t5-small",
            "google/flan-t5-base",
            # Gemma models (may require auth)
            "google/gemma-2b",
            "google/gemma-7b",
            # Mistral models (may require auth)
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            # Llama models (REQUIRES license acceptance on HF website)
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-hf",
        ],
        index=0,
        help="⚠️ Llama models require accepting license on HuggingFace website. Use flan-t5 or gemma-2b for easiest setup.",
    )
    
    if "llama" in local_model.lower():
        st.sidebar.warning(
            "⚠️ Llama models are gated. Make sure you:\n"
            "1. Accepted the license at huggingface.co\n"
            "2. Ran: huggingface-cli login"
        )


# Cache pipeline per model to avoid reloading (models are expensive to load)
@st.cache_resource
def get_pipeline(index_dir, backend, local_model):
    cfg = RagConfig(index_dir=index_dir)
    if backend.startswith("Local"):
        cfg.generator_backend = "local"
        cfg.generator_model = local_model
    else:
        cfg.generator_backend = "openai"
    return RagPipeline(cfg)

pipe = get_pipeline(index_dir, backend, local_model)

if build_clicked:
    with st.spinner("Building index..."):
        # Determine which directory to use
        from src.rag.ingest import load_corpus
        
        if use_uploaded and uploaded_files:
            source_dir = upload_dir
            source_type = "uploaded files"
        else:
            source_dir = docs_dir
            source_type = "directory"
        
        docs = load_corpus(source_dir)
        if len(docs) == 0:
            if use_uploaded and uploaded_files:
                st.error("No uploaded files found! Please upload PDF or TXT files first.")
            elif use_uploaded:
                st.error("No files uploaded yet. Please use the file uploader above.")
            else:
                st.error(f"No documents found in {docs_dir}! Please add .txt or .pdf files or use the file uploader.")
        else:
            st.info(f"Found {len(docs)} document(s) from {source_type}: {', '.join([d['id'] for d in docs])}")
            pipe.build_index(source_dir)
            # Reload the index after building
            pipe.load_index()
            st.success(f"Index built and loaded successfully from {len(docs)} document(s)!")

# Show current index status
try:
    pipe.load_index()
    unique_docs = set(m.get('doc_id', '') for m in pipe.index.chunks)
    st.sidebar.info(f"📚 Index loaded: {len(unique_docs)} document(s)")
    with st.sidebar.expander("View indexed documents"):
        for doc_id in sorted(unique_docs):
            st.write(f"• {doc_id}")
except FileNotFoundError:
    st.sidebar.warning("⚠️ No index found. Build index first!")

question = st.text_input("Ask a question:")
ask = st.button("Ask")

if ask and question.strip():
    try:
        # Ensure index is loaded
        try:
            pipe.load_index()
        except FileNotFoundError:
            st.error("Index not found! Please build the index first by clicking 'Build/Refresh Index'.")
            st.stop()
        
        out = pipe.answer(question, top_k=top_k)
        st.subheader("Answer")
        st.write(out["answer"])

        st.subheader("Context")
        for m in out["passages"]:
            st.markdown(f"**[{m['doc_id']}#{m['chunk_id']}]** (score={m['score']:.3f}) — `{m.get('source_path','')}`")
            st.write(m["text"])
            st.write("---")
    except Exception as e:
        st.error(str(e))



