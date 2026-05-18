import os
import logging
import time
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
HF_EMBED_URL = os.getenv("HF_EMBED_URL")


def get_embedding(text: str):
    if not HF_EMBED_URL:
        raise ValueError("HF_EMBED_URL env var is not set")

    start = time.time()
    logger.debug(f"[EMBED] Requesting embedding for {len(text)} chars")

    response = requests.post(
        HF_EMBED_URL,
        json={"text": text},
        timeout=60,
    )

    response.raise_for_status()
    elapsed = time.time() - start
    logger.debug(f"[EMBED] Done in {elapsed:.2f}s (status {response.status_code})")

    return response.json()["embedding"]