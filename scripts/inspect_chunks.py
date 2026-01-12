import json

CHUNKS_PATH = "data/chunks.json"

def main():
    # Load chunks.json
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Total chunks
    total_chunks = len(chunks)

    # Unique users and meetings
    users = set()
    meetings = set()

    for chunk in chunks:
        users.add(chunk["user_id"])
        meetings.add(chunk["meeting_name"])

    # Print results
    print(f"Total chunks: {total_chunks}")
    print("Users:", ", ".join(sorted(users)))
    print("Meetings:")
    for meeting in sorted(meetings):
        print("-", meeting)

if __name__ == "__main__":
    main()
