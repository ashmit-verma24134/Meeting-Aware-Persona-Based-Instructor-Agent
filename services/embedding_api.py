import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_EMBED_URL = os.getenv("HF_EMBED_URL")

def get_embedding(text: str):

    response = requests.post(
        HF_EMBED_URL,
        json={"text": text}
    )

    response.raise_for_status()

    return response.json()["embedding"]