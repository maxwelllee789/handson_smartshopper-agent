import os
import time

from dotenv import load_dotenv
from google import genai
from google.adk.tools import FunctionTool
from google.genai.errors import ServerError
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from sentence_transformers import SentenceTransformer


load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DB", "smartshopper")
COLLECTION_NAME = os.getenv(
    "MONGODB_COLLECTION",
    "common_information",
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
)

VECTOR_INDEX_NAME = "common_info_vector_index"
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

if not MONGODB_URI:
    raise ValueError("MONGODB_URI belum tersedia di file .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY belum tersedia di file .env")

mongo_client = MongoClient(MONGODB_URI)
database = mongo_client[DATABASE_NAME]
collection = database[COLLECTION_NAME]

print("[INIT] Loading SentenceTransformer...")
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)
print("[INIT] Embedding model ready.")

gemini_client = genai.Client(
    api_key=GOOGLE_API_KEY
)


def retrieve_documents(
    query: str,
    top_k: int = 3,
) -> list:
    """Retrieve relevant Common Information documents with Vector Search."""

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "info_id": 1,
                "category": 1,
                "title": 1,
                "content": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(collection.aggregate(pipeline))


def build_context(documents: list) -> str:
    """Convert retrieved documents into text context for Gemini."""

    context_parts = []

    for index, document in enumerate(documents, start=1):
        context_parts.append(
            f"DOCUMENT {index}\n"
            f"Judul: {document.get('title', '-')}\n"
            f"Kategori: {document.get('category', '-')}\n"
            f"Informasi: {document.get('content', '-')}"
        )

    return "\n\n".join(context_parts)


def generate_answer(
    query: str,
    context: str,
    max_retries: int = 3,
) -> str:
    """Generate a grounded answer with retry for temporary 503 errors."""

    prompt = f"""
Kamu adalah Personalized SmartShopper Assistant.

Jawab pertanyaan user HANYA berdasarkan CONTEXT.
Jangan membuat kebijakan toko yang tidak tersedia di CONTEXT.
Jika context tidak cukup, katakan bahwa informasi tersebut belum tersedia.
Gunakan Bahasa Indonesia yang ringkas dan jelas.
Jangan memberikan product recommendation di tool ini.

CONTEXT:
{context}

PERTANYAAN USER:
{query}
"""

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            if not response.text:
                return "AI tidak menghasilkan jawaban."

            return response.text.strip()

        except ServerError as error:
            error_code = getattr(error, "code", None)

            if error_code == 503:
                if attempt == max_retries - 1:
                    return (
                        "Layanan AI sedang sibuk. "
                        "Silakan coba beberapa saat lagi."
                    )

                wait_time = 2 ** attempt
                print(
                    "[GEMINI] Service unavailable. "
                    f"Retry dalam {wait_time} detik..."
                )
                time.sleep(wait_time)
            else:
                raise


def retrieve_common_information(query: str) -> dict:
    """
    Retrieve and answer general SmartShopper e-commerce information.

    Use this tool for questions about shipping, delivery, tracking,
    purchase process, checkout, payment, cancellation, return, and refund.

    Do not use this tool for personalized product recommendations.

    Args:
        query: User's general e-commerce question.

    Returns:
        Dictionary containing status, answer, and retrieved sources.
    """

    print("\n[COMMON INFO TOOL]")
    print("Query:", query)

    if not query or not query.strip():
        return {
            "status": "error",
            "answer": "Pertanyaan tidak boleh kosong.",
            "sources": [],
        }

    try:
        documents = retrieve_documents(
            query=query.strip(),
            top_k=3,
        )

        if not documents:
            return {
                "status": "not_found",
                "answer": (
                    "Informasi tersebut belum tersedia "
                    "pada SmartShopper."
                ),
                "sources": [],
            }

        print("\n[RETRIEVAL RESULTS]")
        for document in documents:
            print(
                "-",
                document.get("info_id"),
                document.get("title"),
                "score:",
                round(document.get("score", 0), 4),
            )

        context = build_context(documents)
        answer = generate_answer(
            query=query.strip(),
            context=context,
        )

        sources = [
            {
                "info_id": document.get("info_id"),
                "title": document.get("title"),
                "category": document.get("category"),
                "score": float(document.get("score", 0)),
            }
            for document in documents
        ]

        return {
            "status": "success",
            "answer": answer,
            "sources": sources,
        }

    except PyMongoError as error:
        print("[MONGODB ERROR]", error)
        return {
            "status": "error",
            "answer": (
                "Terjadi masalah saat mengambil informasi "
                "dari database."
            ),
            "sources": [],
        }

    except Exception as error:
        print("[COMMON INFO ERROR]", error)
        return {
            "status": "error",
            "answer": "Terjadi kesalahan saat memproses pertanyaan.",
            "sources": [],
        }


common_information_tool = FunctionTool(
    func=retrieve_common_information
)