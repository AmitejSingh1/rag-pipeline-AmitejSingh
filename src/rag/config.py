from dataclasses import dataclass


@dataclass
class RagConfig:
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_words: int = 300
    chunk_overlap_words: int = 60
    top_k: int = 5
    index_dir: str = "indexes"
    metadata_filename: str = "metadata.json"
    faiss_index_filename: str = "faiss.index"
    generator_backend: str = "auto"  # auto | local | openai
    generator_model: str = "google/flan-t5-small"  # default local model



