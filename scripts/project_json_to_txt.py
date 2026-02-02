import json
import os

PROJECT_JSON = "data/project.json"
OUTPUT_TXT = "data/transcripts/cleaned/C_project_project_0.txt"

def main():
    with open(PROJECT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []

    for topic in data.get("topics", []):
        for kf in topic.get("keyframes", []):
            combined = kf.get("combined_summary")
            if combined:
                lines.append(combined.strip())
                lines.append("")  # spacing between summaries

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("combined_summary extracted to:")
    print(OUTPUT_TXT)

if __name__ == "__main__":
    main()
