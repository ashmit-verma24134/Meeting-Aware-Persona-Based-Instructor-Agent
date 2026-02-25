import os
from dotenv import load_dotenv
from gradio_client import Client

load_dotenv()

HF_REPO = os.getenv("HF_REPO")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_REPO:
    raise ValueError("HF_REPO not set in .env")


class HFAPIService:

    def __init__(self):
        if HF_TOKEN:
            self.client = Client(HF_REPO, token=HF_TOKEN)
        else:
            self.client = Client(HF_REPO)

    # ---------------------------------------
    # 1️⃣ Start Pipeline
    # ---------------------------------------
    def start_pipeline(self, video_url: str):

        result = self.client.predict(
            variant="demo-code",
            input_mode="Video URL",
            video_file_path=None,
            video_url=video_url,
            out_dir="",
            python_bin="",
            deepgram_model="nova-3",
            deepgram_language="",
            deepgram_request_timeout_sec=3600,
            deepgram_connect_timeout_sec=30,
            deepgram_retries=3,
            deepgram_retry_backoff_sec=2,
            force_deepgram=False,
            force_keyframes=False,
            pre_roll_sec=3,
            gemini_model="gemini-2.5-flash",
            similarity_threshold=0.82,
            temperature=0.2,
            log_heartbeat_sec=10,
            api_name="/start_pipeline"
        )

        if not isinstance(result, (list, tuple)) or len(result) < 4:
            raise Exception(f"Unexpected HF response: {result}")

        run_id = result[0] or result[3]

        if not run_id:
            raise Exception(f"Could not extract run_id from: {result}")

        return run_id

    # ---------------------------------------
    # 2️⃣ Check Status (Sticky Safe)
    # ---------------------------------------
    def check_status(self, run_id: str):

        result = self.client.predict(
            run_id=run_id,
            tail_lines=0,      # no logs needed
            poll_sec=2,
            api_name="/watch_run"
        )

        status_json = result[0]

        if isinstance(status_json, dict):
            return status_json.get("status", "unknown")

        return "unknown"