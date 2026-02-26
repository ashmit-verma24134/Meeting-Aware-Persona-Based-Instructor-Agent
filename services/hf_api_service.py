import os
from dotenv import load_dotenv
from gradio_client import Client

load_dotenv()

GRADIO_BASE_URL = os.getenv("PIPELINE_BASE_URL")

if not GRADIO_BASE_URL:
    raise ValueError("PIPELINE_BASE_URL not set in .env")


class HFAPIService:

    def __init__(self):
        self.client = Client(GRADIO_BASE_URL)

    # ─────────────────────────────────────
    # 1️⃣ START PIPELINE (GPU VERSION)
    # ─────────────────────────────────────
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
            deepgram_request_timeout_sec=1200,
            deepgram_connect_timeout_sec=30,
            deepgram_retries=3,
            deepgram_retry_backoff_sec=2,
            force_deepgram=False,
            force_keyframes=False,
            pre_roll_sec=3,
            llm_model="llama-3.3-70b-versatile",  # from your API default
            similarity_threshold=0.82,
            temperature=0.2,
            log_heartbeat_sec=10,
            api_name="/start_pipeline"
        )

        if not isinstance(result, (list, tuple)) or len(result) < 4:
            raise Exception(f"Unexpected start response: {result}")

        run_id = result[0]

        if not run_id:
            raise Exception("Could not extract run_id from response")

        return run_id


    # ─────────────────────────────────────
    # 2️⃣ CHECK STATUS
    # ─────────────────────────────────────
# ─────────────────────────────────────
# 2️⃣ CHECK STATUS
# ─────────────────────────────────────
    def check_status(self, run_id: str):

        # 1️⃣ Try checking final output first
        try:
            final_output = self.client.predict(
                rid=run_id,
                api_name="/lambda"
            )

            if isinstance(final_output, dict) and final_output.get("keyframes"):
                return {
                    "status": "completed",
                    "raw": final_output
                }

        except Exception:
            pass  # Not completed yet


        # 2️⃣ Otherwise check logs
        try:
            result = self.client.predict(
                run_id=run_id,
                tail_lines=200,
                api_name="/refresh_status_logs"
            )

            if not isinstance(result, (list, tuple)) or len(result) < 2:
                return {"status": "unknown"}

            status_json = result[0]
            logs = result[1]

            if isinstance(status_json, dict):
                return {
                    "status": status_json.get("status", "running"),
                    "raw": status_json,
                    "logs": logs
                }

            return {"status": "running"}

        except Exception:
            return {"status": "running"}


    # ─────────────────────────────────────
    # 3️⃣ FETCH FINAL OUTPUT
    # ─────────────────────────────────────
    def fetch_result(self, run_id: str):

        result = self.client.predict(
            rid=run_id,
            api_name="/lambda"
        )

        # Case 1: Still running
        if isinstance(result, dict) and result.get("status") == "running":
            return {
                "state": "running",
                "data": result
            }

        # Case 2: Final Output Ready
        if isinstance(result, dict):
            return {
                "state": "completed",
                "data": result
            }

        # Unexpected
        return {
            "state": "unknown",
            "data": result
        }