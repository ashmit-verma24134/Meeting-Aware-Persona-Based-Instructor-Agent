import os
from typing import Dict

TRANSCRIPT_DIR = "data/transcripts/cleaned"


def convert_hf_json_to_txt(
    transcript_json: Dict,
    username: str,
    meeting_name: str
) -> str:

    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR)

    keyframes = transcript_json.get("keyframes", [])

    if not keyframes:
        raise ValueError("Invalid HF transcript format: 'keyframes' not found")

    lines = []

    for frame in keyframes:
        timestamp = frame.get("timestamp", "")
        combined_summary = frame.get("combined_summary", "")

        if combined_summary:
            lines.append(f"[{timestamp}]")
            lines.append(combined_summary.strip())
            lines.append("")

    full_text = "\n".join(lines).strip()

    if not full_text:
        raise ValueError("No combined_summary found in HF output")

    filename = f"C_{username}_{meeting_name}.txt"
    filepath = os.path.join(TRANSCRIPT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"[HF] Transcript saved → {filepath}")

    return filepath