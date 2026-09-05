from pathlib import Path

import pandas as pd
from google.adk.tools import FunctionTool

from recommender import (
    load_model,
    create_product_text,
    create_product_embeddings,
    recommend_products,
)


BASE_DIR = Path(__file__).resolve().parents[1]

users = pd.read_csv(BASE_DIR / "data" / "users.csv")
products = pd.read_csv(BASE_DIR / "data" / "products.csv")
interactions = pd.read_csv(
    BASE_DIR / "data" / "interactions.csv"
)

products = create_product_text(products)

model = load_model()
product_embeddings = create_product_embeddings(
    products,
    model,
)


def get_product_recommendations(
    user_id: str,
    top_n: int = 3,
) -> dict:
    """
    Generate personalized product recommendations for a SmartShopper user.

    Use this tool when the user asks for recommended, suitable,
    or personalized products for a specific user_id such as U001.

    Do not use this tool for shipping, delivery, checkout, payment,
    cancellation, return, or refund questions.

    Args:
        user_id: SmartShopper user identifier, for example U001.
        top_n: Number of recommended products. Maximum 5.

    Returns:
        Dictionary containing personalized product recommendations.
    """

    print("[PRODUCT TOOL]", user_id)

    user_id = user_id.strip().upper()
    top_n = max(1, min(int(top_n), 5))

    try:
        recommendations = recommend_products(
            user_id=user_id,
            users=users,
            interactions=interactions,
            products=products,
            product_embeddings=product_embeddings,
            top_n=top_n,
        )

    except ValueError as error:
        return {
            "status": "error",
            "message": str(error),
        }

    if recommendations.empty:
        return {
            "status": "not_found",
            "message": (
                "Belum cukup data untuk menghasilkan rekomendasi."
            ),
        }

    result = []

    for _, product in recommendations.iterrows():
        result.append(
            {
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "category": product["category"],
                "price": int(product["price"]),
                "semantic_score": float(
                    product["semantic_score"]
                ),
            }
        )

    return {
        "status": "success",
        "user_id": user_id,
        "recommendations": result,
    }


product_recommendation_tool = FunctionTool(
    func=get_product_recommendations
)