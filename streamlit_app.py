import os
import json
import streamlit as st

from src.rag.config import RagConfig
from src.rag.pipeline import RagPipeline


st.set_page_config(page_title="RAG Demo", page_icon="📚", layout="wide")
st.title("RAG Pipeline Demo")

with st.sidebar:
    st.header("Settings")
    docs_dir = st.text_input("Documents directory", value="data")
    index_dir = st.text_input("Index directory", value="indexes")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, step=1)
    build_clicked = st.button("Build/Refresh Index")


cfg = RagConfig(index_dir=index_dir)
pipe = RagPipeline(cfg)

if build_clicked:
    with st.spinner("Building index..."):
        pipe.build_index(docs_dir)
    st.success("Index built.")

question = st.text_input("Ask a question:")
ask = st.button("Ask")

if ask and question.strip():
    try:
        pipe.load_index()
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



