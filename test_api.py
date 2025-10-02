#!/usr/bin/env python3
"""Test OpenRouter API connection."""

import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print(f"API Key loaded: {api_key[:20]}..." if api_key else "No API key found!")

# Test API call
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "qwen/qwen3-14b:free",
    "messages": [
        {"role": "user", "content": "Say hello"}
    ]
}

print("\nTesting OpenRouter API...")
response = requests.post(url, json=data, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
