import os
from typing import List

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def build_prompt(question: str, context_bullets: str) -> str:
    return (
        "You are a helpful assistant. Use ONLY the provided context to answer. "
        "Cite sources like [doc#chunk] inline. If unknown, say you don't know.\n\n"
        f"Question: {question}\n\nContext:\n{context_bullets}\n\nAnswer:"
    )


class LocalGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        # Get HuggingFace token from environment or cache
        from huggingface_hub.utils import HfFolder
        
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if not hf_token:
            # Try to get from cache (from huggingface-cli login)
            try:
                hf_token = HfFolder.get_token()
            except:
                hf_token = None
        
        # Store token for use in from_pretrained calls
        self.hf_token = hf_token
        
        # Try to load tokenizer with token
        try:
            import requests
            
            # First check connectivity
            try:
                requests.get("https://huggingface.co", timeout=5)
            except Exception as conn_err:
                raise ValueError(
                    f"❌ Cannot connect to HuggingFace.co\n\n"
                    f"Network error: {str(conn_err)}\n\n"
                    f"Possible solutions:\n"
                    f"1. Check your internet connection\n"
                    f"2. Check if you're behind a firewall/proxy\n"
                    f"3. Try again in a few minutes (HuggingFace servers might be slow)\n"
                    f"4. If using proxy, set environment variables:\n"
                    f"   $env:HTTP_PROXY = 'your_proxy'\n"
                    f"   $env:HTTPS_PROXY = 'your_proxy'"
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token if hf_token else True  # Use token if available, otherwise try to use cached login
            )
        except ValueError as ve:
            # Re-raise our custom network errors
            raise
        except Exception as e:
            error_msg = str(e)
            if "couldn't connect" in error_msg.lower() or "connection" in error_msg.lower() or "we couldn't connect" in error_msg.lower():
                raise ValueError(
                    f"❌ Network connection error\n\n"
                    f"Error: {error_msg}\n\n"
                    f"Troubleshooting:\n"
                    f"1. Check your internet connection\n"
                    f"2. Try again in a few minutes (HuggingFace servers might be slow)\n"
                    f"3. Check firewall/proxy settings\n"
                    f"4. If using corporate network, you may need to configure proxy:\n"
                    f"   $env:HTTP_PROXY = 'http://proxy:port'\n"
                    f"   $env:HTTPS_PROXY = 'http://proxy:port'\n"
                    f"5. Try downloading the model manually first:\n"
                    f"   huggingface-cli download {model_name}"
                )
            elif "gated" in error_msg.lower() or "restricted" in error_msg.lower() or "401" in error_msg:
                # Provide helpful error message with token status
                token_status = "found" if self.hf_token else "not found"
                raise ValueError(
                    f"❌ Model {model_name} is gated/restricted.\n\n"
                    f"Token status: {token_status}\n\n"
                    f"To fix this:\n"
                    f"1. Visit https://huggingface.co/{model_name} in your browser\n"
                    f"2. Click 'Agree and access repository' to accept the license (if not done)\n"
                    f"3. Make sure you're logged in: huggingface-cli login\n"
                    f"4. If still failing, set token manually: $env:HUGGINGFACE_HUB_TOKEN = 'your_token'\n"
                    f"5. Get your token from: https://huggingface.co/settings/tokens\n"
                    f"6. Restart Streamlit app\n\n"
                    f"Original error: {error_msg}"
                )
            else:
                raise ValueError(
                    f"Failed to load tokenizer for {model_name}. "
                    f"Error: {error_msg}"
                )
        
        # Try to detect model type - decoder-only (causal) vs encoder-decoder (seq2seq)
        # Common decoder-only models: llama, mistral, gemma, phi, etc.
        is_decoder_only = any(x in model_name.lower() for x in [
            "llama", "mistral", "gemma", "phi", "qwen", "falcon", "mpt"
        ])
        
        if is_decoder_only:
            # Use CausalLM for decoder-only models
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=self.hf_token if self.hf_token else True
                )
            except Exception as e:
                error_msg = str(e)
                if "gated" in error_msg.lower() or "restricted" in error_msg.lower() or "401" in error_msg:
                    raise ValueError(
                        f"❌ Model {model_name} is gated/restricted.\n\n"
                        f"To access it:\n"
                        f"1. Visit https://huggingface.co/{model_name} in your browser\n"
                        f"2. Click 'Agree and access repository' to accept the license\n"
                        f"3. Make sure you're logged in: huggingface-cli login\n"
                        f"4. Refresh the Streamlit app\n\n"
                        f"Original error: {error_msg}"
                    )
                else:
                    raise ValueError(f"Failed to load model {model_name}. Error: {error_msg}")
            # Set pad_token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.is_causal = True
        else:
            # Use Seq2SeqLM for encoder-decoder models (flan-t5, etc.)
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    token=self.hf_token if self.hf_token else True
                )
            except Exception as e:
                raise ValueError(f"Failed to load model {model_name}. Error: {e}")
            self.is_causal = False

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.is_causal:
            # For decoder-only models, use text generation pipeline
            pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            outputs = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return outputs[0]["generated_text"].strip()
        else:
            # For encoder-decoder models, use text2text generation
            pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
            out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
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



