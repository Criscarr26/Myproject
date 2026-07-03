# Recomendador de productos

Sistema de recomendación de productos en Python usando **filtrado colaborativo
item-item**: a partir del historial de compras se construye una matriz
cliente-producto, se calcula la **similitud coseno** entre productos y se
recomiendan los más parecidos a los que el cliente ya compró.

## Cómo funciona

1. El historial de compras se convierte en una matriz cliente x producto
   (pivot table con pandas).
2. Se calcula la similitud coseno entre las columnas (productos): dos productos
   son similares si los compran los mismos clientes.
3. Para un cliente dado, se suman las similitudes de cada producto con todo lo
   que ya compró, se excluyen sus compras previas y se devuelven los 5 mejores.

## Uso

```bash
pip install -r requirements.txt
python recomendaciones.py
```

El script incluye un historial de ejemplo y muestra las recomendaciones para
cada cliente. Para usar datos propios, reemplaza `DATA` con registros
`{"customer_id": ..., "product_id": ...}`.

```python
from recomendaciones import build_matrices, suggest_products

customer_product, similarity_df = build_matrices(mis_datos)
print(suggest_products(cliente_id, customer_product, similarity_df))
```

## Licencia

[MIT](LICENSE)
