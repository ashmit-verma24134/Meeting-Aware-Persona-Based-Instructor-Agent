import json
from scripts.hf_json_to_txt import convert_hf_json_to_txt

JSON_PATH = "sample.json"
def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_path = convert_hf_json_to_txt(
        transcript_json=data,
        username="test_user",
        meeting_name="test_meeting"
    )

    print("✅ Output saved at:", output_path)

if __name__ == "__main__":
    main()