"""Quick test to verify HuggingFace token and access"""
import os
from huggingface_hub.utils import HfFolder
from transformers import AutoTokenizer

print("=" * 60)
print("Testing HuggingFace Token and Access")
print("=" * 60)

# Get token
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        hf_token = HfFolder.get_token()
        if hf_token:
            print(f"[OK] Found token from cache (huggingface-cli login)")
        else:
            print("[ERROR] Token is None in cache")
            hf_token = None
    except Exception as e:
        print(f"[ERROR] Could not get token from cache: {e}")
        hf_token = None
else:
    print(f"[OK] Found token in environment")

if not hf_token:
    print("[ERROR] No token found!")
    print("Run: huggingface-cli login")
    print("Or set: $env:HUGGINGFACE_HUB_TOKEN = 'your_token'")
    print("Get token from: https://huggingface.co/settings/tokens")
    exit(1)

print(f"\nToken (first 10 chars): {hf_token[:10] if hf_token else 'None'}...")

# Test with gemma-2b
model_name = "google/gemma-2b"
print(f"\nTesting access to: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    print(f"[SUCCESS] ✓ Successfully loaded tokenizer for {model_name}!")
    print(f"Vocab size: {len(tokenizer)}")
except Exception as e:
    error_msg = str(e)
    if "401" in error_msg or "gated" in error_msg.lower():
        print(f"[ERROR] ✗ Access denied: {error_msg}")
        print("\nTroubleshooting:")
        print("1. Make sure you accepted the license at:")
        print(f"   https://huggingface.co/{model_name}")
        print("2. Wait 2-3 minutes after accepting")
        print("3. Try logging in again: huggingface-cli login")
        print("4. Or set token manually:")
        print("   $env:HUGGINGFACE_HUB_TOKEN = 'your_token_here'")
        print("   Get token from: https://huggingface.co/settings/tokens")
    else:
        print(f"[ERROR] Unexpected error: {e}")

