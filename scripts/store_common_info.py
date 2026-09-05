import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DB", "smartshopper")
COLLECTION_NAME = os.getenv(
    "MONGODB_COLLECTION",
    "common_information",
)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "common_information.json"

with open(DATA_FILE, "r", encoding="utf-8") as file:
    documents = json.load(file)

print(f"Loaded {len(documents)} documents.")

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Embedding model ready.")

client = MongoClient(MONGODB_URI)
database = client[DATABASE_NAME]
collection = database[COLLECTION_NAME]

collection.create_index("info_id", unique=True)

for document in documents:
    text_for_embedding = (
        f"Kategori: {document['category']}. "
        f"Judul: {document['title']}. "
        f"Informasi: {document['content']}. "
        f"Keywords: {' '.join(document['keywords'])}"
    )

    embedding = model.encode(
        text_for_embedding,
        normalize_embeddings=True,
    ).tolist()

    mongo_document = {
        **document,
        "embedding_text": text_for_embedding,
        "embedding": embedding,
    }

    collection.update_one(
        {"info_id": document["info_id"]},
        {"$set": mongo_document},
        upsert=True,
    )

    print(
        f"Stored {document['info_id']} - {document['title']}"
    )

print("\nFinished storing Common Information.")