from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

customer_ids = [1, 2, 3]
feature_vector = store.get_online_features(
    features=[
        "customer_features:transaction_count",
        "customer_features:total_spent"
    ],
    entity_rows=[{"customer_id": cid} for cid in customer_ids]
)

df = pd.DataFrame.from_dict(feature_vector.to_dict())
df = df[["customer_id", "transaction_count", "total_spent"]]
print(df)