"""Helper script to check HuggingFace authentication and access to gated models"""
import os
import sys
from huggingface_hub import login, whoami, HfApi

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("HuggingFace Authentication Check")
print("=" * 60)

# Check if token is in environment
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if hf_token:
    print(f"[OK] Found HF token in environment")
    try:
        login(token=hf_token, add_to_git_credential=False)
    except:
        pass
else:
    print("[INFO] No HF token in environment variables")
    print("  Token might be in cache from 'huggingface-cli login'")
    print("  If needed, you can set it with: $env:HUGGINGFACE_HUB_TOKEN = 'your_token'")
    
    # Try to use cached token
    try:
        from huggingface_hub.utils import HfFolder
        cached_token = HfFolder.get_token()
        if cached_token:
            print(f"[OK] Found cached token from huggingface-cli login")
            login(token=cached_token, add_to_git_credential=False)
    except:
        pass

# Check current login status
try:
    user_info = whoami()
    print(f"[OK] Logged in as: {user_info.get('name', 'Unknown')}")
except Exception as e:
    print(f"[ERROR] Not logged in or token invalid: {e}")
    print("\nTo login, run:")
    print("  huggingface-cli login")
    exit(1)

# Check access to specific models
api = HfApi()
test_models = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "google/gemma-2b",
    "mistralai/Mistral-7B-v0.1",
]

print("\n" + "=" * 60)
print("Checking access to models:")
print("=" * 60)

for model in test_models:
    try:
        model_info = api.model_info(model)
        is_gated = model_info.gated if hasattr(model_info, 'gated') else False
        if is_gated:
            print(f"[GATED] {model}: Requires license acceptance")
            print(f"   -> Visit: https://huggingface.co/{model}")
            print(f"   -> Click 'Agree and access repository'")
        else:
            print(f"[OK] {model}: Accessible")
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower():
            print(f"[NO ACCESS] {model}: License not accepted")
            print(f"   -> Visit: https://huggingface.co/{model}")
            print(f"   -> Click 'Agree and access repository'")
        else:
            print(f"[?] {model}: {e}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("If you see 'NO ACCESS' for Llama models:")
print("1. Go to https://huggingface.co/meta-llama/Llama-2-7b-hf")
print("2. Click 'Agree and access repository'")
print("3. Wait a few minutes for access to propagate")
print("4. Restart your Streamlit app")

