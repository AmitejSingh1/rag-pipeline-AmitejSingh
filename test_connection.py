"""Test HuggingFace connectivity"""
import requests
import sys

print("Testing connection to HuggingFace...")
print("=" * 60)

try:
    print("Testing basic connectivity...")
    response = requests.get("https://huggingface.co", timeout=10)
    print(f"[OK] Can reach huggingface.co (status: {response.status_code})")
except requests.exceptions.Timeout:
    print("[ERROR] Connection timeout - HuggingFace servers might be slow or unreachable")
    print("Try again in a few minutes")
    sys.exit(1)
except requests.exceptions.ConnectionError as e:
    print(f"[ERROR] Cannot connect to HuggingFace.co")
    print(f"Error: {e}")
    print("\nPossible causes:")
    print("1. No internet connection")
    print("2. Firewall blocking the connection")
    print("3. Corporate proxy/VPN issue")
    print("4. DNS resolution problem")
    print("\nTroubleshooting:")
    print("- Try opening https://huggingface.co in your browser")
    print("- If behind proxy, set environment variables:")
    print("  $env:HTTP_PROXY = 'http://proxy:port'")
    print("  $env:HTTPS_PROXY = 'http://proxy:port'")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    sys.exit(1)

# Test API endpoint
try:
    print("\nTesting HuggingFace API endpoint...")
    response = requests.get("https://huggingface.co/api/models", timeout=10)
    print(f"[OK] Can reach HuggingFace API (status: {response.status_code})")
except Exception as e:
    print(f"[WARN] API endpoint test failed: {e}")
    print("This might be okay, try downloading a model anyway")

print("\n" + "=" * 60)
print("[SUCCESS] Connection test passed!")
print("You should be able to download models now.")

