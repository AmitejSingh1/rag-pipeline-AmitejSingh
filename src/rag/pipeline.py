import json
import os
from typing import Dict, List

import numpy as np

from .config import RagConfig
from .ingest import load_corpus
from .chunk import chunk_documents
from .embeddings import EmbeddingModel
from .index_faiss import FaissIndex
from .retriever import format_context
from .generator import LocalGenerator, OpenAIGenerator, build_prompt


class RagPipeline:
    def __init__(self, config: RagConfig):
        self.cfg = config
        self.embedder = EmbeddingModel(config.embed_model_name)
        self.index = FaissIndex(
            index_dir=config.index_dir,
            faiss_index_filename=config.faiss_index_filename,
            metadata_filename=config.metadata_filename,
        )

        # Select generator backend
        backend = config.generator_backend
        if backend == "auto":
            if os.environ.get("OPENAI_API_KEY"):
                backend = "openai"
            else:
                backend = "local"
        if backend == "openai":
            self.generator = OpenAIGenerator(model="gpt-4o-mini")
        else:
            self.generator = LocalGenerator(model_name=config.generator_model)

    def build_index(self, docs_dir: str):
        docs = load_corpus(docs_dir)
        chunks = chunk_documents(docs, self.cfg.chunk_size_words, self.cfg.chunk_overlap_words)
        vectors = self.embedder.encode([c["text"] for c in chunks])
        self.index.build(vectors, chunks)
        self.index.save()

    def load_index(self):
        self.index.load()

    def retrieve(self, question: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or self.cfg.top_k
        q_vec = self.embedder.encode([question])[0]
        return self.index.search(q_vec, top_k=top_k)

    def answer(self, question: str, top_k: int = None) -> Dict:
        passages = self.retrieve(question, top_k)
        context_bullets = format_context(passages)
        prompt = build_prompt(question, context_bullets)
        answer = self.generator.generate(prompt)
        return {"question": question, "answer": answer, "passages": passages}



