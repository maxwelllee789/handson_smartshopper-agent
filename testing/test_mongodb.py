import os

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")

print("URI loaded:", bool(mongodb_uri))

client = MongoClient(mongodb_uri)
client.admin.command("ping")

print("MongoDB connected successfully!")   