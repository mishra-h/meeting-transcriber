# test_token.py
from config import HUGGING_FACE_TOKEN
from huggingface_hub import HfApi

print(f"Token: {HUGGING_FACE_TOKEN[:10]}...")

try:
    api = HfApi()
    user = api.whoami(token=HUGGING_FACE_TOKEN)
    print(f"✅ Token is valid! Logged in as: {user['name']}")
except Exception as e:
    print(f"❌ Token error: {e}")
