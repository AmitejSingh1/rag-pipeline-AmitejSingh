from src.rag.chunk import chunk_text_words


def test_chunk_text_words_basic():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text_words(text, chunk_size_words=4, overlap_words=2)
    assert len(chunks) >= 3
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)



