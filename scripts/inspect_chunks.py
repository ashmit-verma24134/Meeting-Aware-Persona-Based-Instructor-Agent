#simple debugging tool for seeing the chunking results 

import json

CHUNKS_PATH = "data/chunks.json"

def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    total_chunks = len(chunks)

    users = set()
    meetings = set()

    for chunk in chunks:
        users.add(chunk["user_id"])
        meetings.add(chunk["meeting_name"])

    print(f"Total chunks: {total_chunks}")
    print("Users:", ", ".join(sorted(users)))
    print("Meetings:")
    for meeting in sorted(meetings):
        print("-", meeting)

if __name__ == "__main__":
    main()