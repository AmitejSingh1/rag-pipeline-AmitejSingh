import os
from typing import Dict, Iterable, List
from pypdf import PdfReader


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    pages.append(text)
            except Exception as e:
                print(f"Warning: Could not extract text from page {i+1} of {file_path}: {e}")
        if not pages:
            raise ValueError(f"No text could be extracted from PDF: {file_path}")
        return "\n".join(pages)
    except Exception as e:
        raise ValueError(f"Failed to read PDF {file_path}: {e}")


def iter_documents(docs_dir: str) -> Iterable[Dict]:
    for root, _, files in os.walk(docs_dir):
        for name in files:
            lower = name.lower()
            path = os.path.join(root, name)
            if lower.endswith(".txt"):
                try:
                    text = load_txt(path)
                    if not text.strip():
                        print(f"Warning: {path} is empty, skipping")
                        continue
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            elif lower.endswith(".pdf"):
                try:
                    text = load_pdf(path)
                    if not text.strip():
                        print(f"Warning: {path} appears empty (no text extracted), skipping")
                        continue
                except Exception as e:
                    print(f"Error loading PDF {path}: {e}")
                    continue
            else:
                continue
            yield {"id": os.path.relpath(path, docs_dir), "source_path": path, "text": text}


def load_corpus(docs_dir: str) -> List[Dict]:
    return list(iter_documents(docs_dir))



