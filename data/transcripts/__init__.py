# data/transcripts/__init__.py

import os

BASE_TRANSCRIPTS_DIR = os.path.dirname(__file__)


def load_full_transcript(user_id, meeting_name):
    """
    Load and concatenate the full transcript for a given meeting.
    Assumes transcript files are stored as text files.
    """

    transcript_texts = []

    for filename in os.listdir(BASE_TRANSCRIPTS_DIR):
        if meeting_name in filename:
            file_path = os.path.join(BASE_TRANSCRIPTS_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                transcript_texts.append(f.read())

    if not transcript_texts:
        raise FileNotFoundError(
            f"No transcript files found for meeting: {meeting_name}"
        )

    return "\n\n".join(transcript_texts)
