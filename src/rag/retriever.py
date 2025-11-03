from typing import Dict, List


def format_context(passages: List[Dict]) -> str:
    lines = []
    for p in passages:
        tag = f"[{p['doc_id']}#{p['chunk_id']}]"
        lines.append(f"- {tag} {p['text']}")
    return "\n".join(lines)



