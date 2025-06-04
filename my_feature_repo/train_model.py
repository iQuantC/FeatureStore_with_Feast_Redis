import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_parquet("data/customer_transactions.parquet")
df = df.sort_values(by=["event_timestamp"]).drop_duplicates("customer_id", keep="last")

df["churn"] = (df["transaction_count"] < 31).astype(int)

X = df[["transaction_count", "total_spent"]]
y = df["churn"]
print(df["churn"].value_counts())

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")