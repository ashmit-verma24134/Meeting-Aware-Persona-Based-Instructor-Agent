import os
from typing import Dict

# Folder where transcripts are stored
TRANSCRIPT_DIR = "data/transcripts/cleaned"


def convert_hf_json_to_txt(
    transcript_json: Dict,
    username: str,
    meeting_name: str
) -> str:
    """
    Converts HF API JSON transcript into your internal .txt format.

    Expected JSON structure:
    {
        "topics": [
            {
                "topic": "...",
                "keyframes": [
                    {
                        "keyframe": {
                            "timestamp": "00:00:00"
                        },
                        "combined_summary": "...",
                        "changed_summary": "..."
                    }
                ]
            }
        ]
    }

    Output format:
    C_username_meetingname.txt
    """

    # Ensure transcript directory exists
    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR)

    lines = []

    topics = transcript_json.get("topics", [])

    if not topics:
        raise ValueError("Invalid HF transcript format: 'topics' not found")

    for topic in topics:
        topic_title = topic.get("topic", "Unknown Topic")

        lines.append(f"\n==============================")
        lines.append(f"Topic: {topic_title}")
        lines.append(f"==============================\n")

        keyframes = topic.get("keyframes", [])

        for kf in keyframes:

            keyframe_data = kf.get("keyframe", {})
            timestamp = keyframe_data.get("timestamp", "")

            combined_summary = kf.get("combined_summary", "")
            changed_summary = kf.get("changed_summary")

            if combined_summary:
                lines.append(f"[{timestamp}]")
                lines.append(combined_summary.strip())
                lines.append("")

            # Optional: include transition summary if exists
            if changed_summary:
                lines.append(f"[{timestamp} - Transition]")
                lines.append(changed_summary.strip())
                lines.append("")

    full_text = "\n".join(lines).strip()

    if not full_text:
        raise ValueError("Transcript text extraction failed")

    # Create filename in your required format
    filename = f"C_{username}_{meeting_name}.txt"
    filepath = os.path.join(TRANSCRIPT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"[HF] Transcript saved → {filepath}")

    return filepath
