from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["voice_agent_db"]
collection = db["call_requests"]

phone = input("Enter phone number: ")

record = collection.find_one({"phone": phone})

if record:
    print("\n--- CALL TRANSCRIPT ---\n")
    print(record.get("call_transcript", "No transcript found"))
else:
    print("No record found")
