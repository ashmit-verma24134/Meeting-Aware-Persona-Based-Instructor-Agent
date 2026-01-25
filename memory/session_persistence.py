import json
from pathlib import Path
from datetime import datetime
import re


def _safe_filename(name: str) -> str:
    """Remove unsafe characters from filenames."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def save_session(session_filename: str, session_data: dict):
    """
    Save a completed session to a JSON file.

    session_filename:
        <user_id>_<meeting_name>_<timestamp>.json

    session_data format:
    {
        "user_id": str,
        "meeting_name": str,
        "session_start_time": str,
        "session_end_time": str,
        "conversation": [
            {
                "question": str,
                "answer": str,
                "timestamp": str
            }
        ]
    }
    """

    sessions_dir = Path("sessions")
    sessions_dir.mkdir(exist_ok=True)

    # 🔒 Sanitize filename
    safe_name = _safe_filename(session_filename)

    # 🕒 Add fallback timestamp if missing
    if not safe_name.endswith(".json"):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{safe_name}_{ts}.json"

    file_path = sessions_dir / safe_name

    # 🛡 Atomic write (write → rename)
    temp_path = file_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    temp_path.replace(file_path)

    print(f"[SessionPersistence] Session saved → {file_path}")
