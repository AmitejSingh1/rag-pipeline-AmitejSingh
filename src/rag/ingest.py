import os
from typing import Dict, Iterable, List
from pypdf import PdfReader


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def iter_documents(docs_dir: str) -> Iterable[Dict]:
    for root, _, files in os.walk(docs_dir):
        for name in files:
            lower = name.lower()
            path = os.path.join(root, name)
            if lower.endswith(".txt"):
                text = load_txt(path)
            elif lower.endswith(".pdf"):
                text = load_pdf(path)
            else:
                continue
            yield {"id": os.path.relpath(path, docs_dir), "source_path": path, "text": text}


def load_corpus(docs_dir: str) -> List[Dict]:
    return list(iter_documents(docs_dir))



