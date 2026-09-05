# testing/test_recommender.py
import pandas as pd

from recommender import (
    load_model,
    create_product_text,
    create_product_embeddings,
    recommend_products,
)

users = pd.read_csv("data/users.csv")
products = pd.read_csv("data/products.csv")
interactions = pd.read_csv("data/interactions.csv")

model = load_model()
products = create_product_text(products)
product_embeddings = create_product_embeddings(products, model)

recommendations = recommend_products(
    user_id="U001",
    users=users,
    interactions=interactions,
    products=products,
    product_embeddings=product_embeddings,
    top_n=3,
)

print(
    recommendations[[
        "product_id",
        "product_name",
        "price",
        "semantic_score",
    ]]
)