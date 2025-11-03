from typing import Dict, List


def chunk_text_words(text: str, chunk_size_words: int, overlap_words: int) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size_words)
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start = max(0, end - overlap_words)
    return chunks


def chunk_documents(docs: List[Dict], chunk_size_words: int, overlap_words: int) -> List[Dict]:
    out: List[Dict] = []
    for d in docs:
        chunks = chunk_text_words(d.get("text", ""), chunk_size_words, overlap_words)
        for i, ch in enumerate(chunks):
            out.append({
                "doc_id": d["id"],
                "chunk_id": i,
                "text": ch,
                "source_path": d.get("source_path"),
            })
    return out



