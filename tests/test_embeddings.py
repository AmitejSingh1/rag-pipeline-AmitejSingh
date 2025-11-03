from src.rag.embeddings import EmbeddingModel


def test_embeddings_shape():
    model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    vecs = model.encode(["hello world", "rag pipeline"])
    assert vecs.shape[0] == 2
    assert vecs.shape[1] > 10



