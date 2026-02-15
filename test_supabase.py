import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

print("URL:", SUPABASE_URL)
print("KEY loaded:", SUPABASE_KEY is not None)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Try inserting a dummy meeting
response = supabase.table("meetings").insert({
    "meeting_name": "test_meeting"
}).execute()

print("Insert response:", response)
