"""Sistema de recomendación de productos con filtrado colaborativo.

Construye una matriz cliente-producto a partir del historial de compras,
calcula la similitud coseno entre productos y sugiere al cliente los
productos más parecidos a los que ya compró.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Historial de compras de ejemplo (cliente -> producto)
DATA = [
    {"customer_id": 1, "product_id": 101},
    {"customer_id": 1, "product_id": 102},
    {"customer_id": 2, "product_id": 101},
    {"customer_id": 2, "product_id": 103},
    {"customer_id": 3, "product_id": 104},
    {"customer_id": 3, "product_id": 105},
]


def build_matrices(data):
    """Devuelve la matriz cliente-producto y la de similitud entre productos."""
    df = pd.DataFrame(data)
    customer_product = df.pivot_table(
        index="customer_id", columns="product_id", aggfunc=len, fill_value=0
    )
    similarity = cosine_similarity(customer_product.T)
    similarity_df = pd.DataFrame(
        similarity,
        index=customer_product.columns,
        columns=customer_product.columns,
    )
    return customer_product, similarity_df


def suggest_products(customer_id, customer_product, similarity_df, top_n=5):
    """Sugiere los top_n productos más similares a los ya comprados por el cliente."""
    row = customer_product.loc[customer_id]
    purchased = row[row > 0].index

    # Suma la similitud de cada producto con todo lo que el cliente ya compró
    scores = similarity_df.loc[purchased].sum(axis=0)

    # Excluye únicamente los productos ya comprados
    scores = scores[~scores.index.isin(purchased)]

    return scores.sort_values(ascending=False).head(top_n)


if __name__ == "__main__":
    customer_product, similarity_df = build_matrices(DATA)
    for customer in customer_product.index:
        print(f"Recomendaciones para el cliente {customer}:")
        print(suggest_products(customer, customer_product, similarity_df))
        print()
