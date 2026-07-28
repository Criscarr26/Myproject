# Product recommender

Product recommendation system in Python using **item-item collaborative
filtering**: from purchase history it builds a customer-product matrix,
computes the **cosine similarity** between products and recommends the ones
most similar to what the customer already bought.

## How it works

1. The purchase history is turned into a customer x product matrix
   (pivot table with pandas).
2. Cosine similarity is computed between the columns (products): two products
   are similar if the same customers buy them.
3. For a given customer, the similarities of every product with everything they
   already bought are summed, previous purchases are excluded and the top 5 are returned.

## Usage

```bash
pip install -r requirements.txt
python recomendaciones.py
```

The script includes a sample history and prints the recommendations for each
customer. To use your own data, replace `DATA` with
`{"customer_id": ..., "product_id": ...}` records.

```python
from recomendaciones import build_matrices, suggest_products

customer_product, similarity_df = build_matrices(my_data)
print(suggest_products(customer_id, customer_product, similarity_df))
```

## License

[MIT](LICENSE)
