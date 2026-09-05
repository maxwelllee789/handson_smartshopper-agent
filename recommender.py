import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def load_model():
    return SentenceTransformer(MODEL_NAME)


def create_product_text(products):
    products = products.copy()

    products["product_text"] = (
        "Nama produk: " + products["product_name"] + ". "
        "Kategori: " + products["category"] + ". "
        "Warna: " + products["color"] + ". "
        "Style: " + products["style"] + ". "
        "Fit: " + products["fit"] + ". "
        "Occasion: " + products["occasion"] + ". "
        "Deskripsi: " + products["description"]
    )

    return products


def create_product_embeddings(products, model):
    return model.encode(
        products["product_text"].tolist(),
        normalize_embeddings=True,
    )


INTERACTION_WEIGHTS = {
    "click": 1.0,
    "wishlist": 2.0,
    "purchase": 3.0,
}


def create_user_embedding(
    user_id,
    interactions,
    products,
    product_embeddings,
):
    user_history = interactions[
        interactions["user_id"] == user_id
    ].copy()

    if user_history.empty:
        return None

    product_index = {
        product_id: index
        for index, product_id in enumerate(products["product_id"])
    }

    embeddings = []
    weights = []

    for _, interaction in user_history.iterrows():
        product_id = interaction["product_id"]
        interaction_type = interaction["interaction_type"]

        if product_id not in product_index:
            continue

        index = product_index[product_id]
        embeddings.append(product_embeddings[index])
        weights.append(
            INTERACTION_WEIGHTS.get(interaction_type, 1.0)
        )

    if not embeddings:
        return None

    user_embedding = np.average(
        np.array(embeddings),
        axis=0,
        weights=np.array(weights),
    )

    norm = np.linalg.norm(user_embedding)
    if norm != 0:
        user_embedding = user_embedding / norm

    return user_embedding


def recommend_products(
    user_id,
    users,
    interactions,
    products,
    product_embeddings,
    top_n=5,
):
    if user_id not in users["user_id"].values:
        raise ValueError(f"User {user_id} tidak ditemukan.")

    user_embedding = create_user_embedding(
        user_id=user_id,
        interactions=interactions,
        products=products,
        product_embeddings=product_embeddings,
    )

    if user_embedding is None:
        return pd.DataFrame()

    similarities = cosine_similarity(
        [user_embedding],
        product_embeddings,
    )[0]

    result = products.copy()
    result["semantic_score"] = similarities

    seen_products = interactions[
        interactions["user_id"] == user_id
    ]["product_id"].unique()

    result = result[
        ~result["product_id"].isin(seen_products)
    ]

    user = users[
        users["user_id"] == user_id
    ].iloc[0]

    result = result[
        result["price"] <= user["max_budget"]
    ]

    result = result.sort_values(
        by="semantic_score",
        ascending=False,
    )

    return result.head(top_n)