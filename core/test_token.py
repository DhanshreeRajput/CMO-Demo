import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACE_API_KEY")

print("🔍 Direct API Test:")
print(f"Token: {token[:10]}...{token[-4:]}")

# Test 1: Check token validity with HF API
headers = {"Authorization": f"Bearer {token}"}

try:
    # Test with user info endpoint
    response = requests.get("https://huggingface.co/api/whoami", headers=headers)
    print(f"\n🧪 User Info Test:")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        user_info = response.json()
        print(f"✅ Token is valid!")
        print(f"Username: {user_info.get('name', 'N/A')}")
        print(f"Type: {user_info.get('type', 'N/A')}")
    else:
        print(f"❌ Token invalid: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")

# Test 2: Try inference with different model
try:
    print(f"\n🧪 Inference Test:")
    
    # Use a very simple model for testing
    inference_url = "https://api-inference.huggingface.co/models/gpt2"
    
    payload = {
        "inputs": "Hello",
        "parameters": {"max_length": 10}
    }
    
    response = requests.post(inference_url, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Inference API works!")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Inference failed: {response.text}")
        
except Exception as e:
    print(f"❌ Inference test failed: {e}")