import os
from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def build_prompt(question: str, context_bullets: str) -> str:
    return (
        "You are a helpful assistant. Use ONLY the provided context to answer. "
        "Cite sources like [doc#chunk] inline. If unknown, say you don't know.\n\n"
        f"Question: {question}\n\nContext:\n{context_bullets}\n\nAnswer:"
    )


class LocalGenerator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"].strip()


class OpenAIGenerator:
    def __init__(self, model: str):
        from openai import OpenAI  # lazy import

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 400) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()



