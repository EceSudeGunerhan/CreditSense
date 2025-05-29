import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("HF_API_URL")
API_KEY = os.getenv("HF_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

def ask_llm(prompt: str):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"API error: {response.status_code} - {response.text}"
