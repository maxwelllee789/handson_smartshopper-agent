import os

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
INDEX_NAME = "common_info_vector_index"

client = MongoClient(MONGODB_URI)
collection = client[DATABASE_NAME][COLLECTION_NAME]

model = SentenceTransformer(
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

query = (
    "Gimana caranya balikin barang "
    "dan dapat uang kembali?"
)

query_embedding = model.encode(
    query,
    normalize_embeddings=True,
).tolist()

pipeline = [
    {
        "$vectorSearch": {
            "index": INDEX_NAME,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 50,
            "limit": 3,
        }
    },
    {
        "$project": {
            "_id": 0,
            "info_id": 1,
            "title": 1,
            "category": 1,
            "content": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
    },
]

results = list(collection.aggregate(pipeline))

for result in results:
    print("\n----------------------")
    print("ID:", result["info_id"])
    print("Title:", result["title"])
    print("Score:", result["score"])
    print("Content:", result["content"])