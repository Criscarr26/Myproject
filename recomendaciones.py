data = [
    {"customer_id": 1, "product_id": 101},
    {"customer_id": 1, "product_id": 102},
    {"customer_id": 2, "product_id": 101},
    {"customer_id": 2, "product_id": 103},
    {"customer_id": 3, "product_id": 104},
    {"customer_id": 3, "product_id": 105},
    ]
import pandas as pd

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Create a pivot table to get the customer-product matrix
customer_product= df.pivot_table(index='customer_id', columns='product_id', aggfunc=len, fill_value=0)
from sklearn.metrics.pairwise import cosine_similarity

# Calculate product similarity
product_similarity = cosine_similarity(customer_product.T)

# Convert the similarity matrix into a DataFrame
similarity_df = pd.DataFrame(product_similarity, index=customer_product.columns, columns=customer_product.columns)
def suggest_products(customer_id, customer_product, similarity_df):
  purchased_products = customer_product.loc[customer_id]
  similar_products = similarity_df.loc[purchased_products.index]

  # Filter out products that have already been purchased
  recommendations = similar_products.sum(axis=1).sort_values(ascending=False)
  recommendations = recommendations[~recommendations.index.isin(purchased_products.index)]

  return recommendations.head(5)  # Suggest the top 5 most similar products

# Example: suggestions for customer 1
print(suggest_products(1, customer_product, similarity_df))

