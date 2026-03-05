import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API")
if not api_key:
    raise RuntimeError("GEMINI_API is not set. Add it to your .env file.")

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

headers = {
    "Content-Type": "application/json",
    "X-goog-api-key": api_key,
}

data = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Explain how AI works in a few words"
                }
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=data, timeout=30)
response.raise_for_status()

print(response.status_code)
print(response.json())