import streamlit as st
import pandas as pd
import joblib
from feast import FeatureStore
import plotly.graph_objects as go

store = FeatureStore(repo_path="feature_repo")
model = joblib.load("model.pkl")

st.title("📊 Customer Churn Predictor")

# Input
customer_id = st.text_input("Enter Customer ID:", value="1")

# Validate input
if not customer_id.isdigit() or int(customer_id) <= 0:
    st.warning("⚠️ Please enter a valid positive integer for Customer ID.")
    st.stop()

customer_id = int(customer_id)

# Fetch features
features = store.get_online_features(
    features=[
        "customer_features:transaction_count",
        "customer_features:total_spent",
    ],
    entity_rows=[{"customer_id": customer_id}],
).to_df()

df = pd.DataFrame.from_dict(features.to_dict())

# Show retrieved features
st.subheader("🔍 Retrieved Features")
st.dataframe(df)

if features.isnull().values.any():
    st.error(f"❌ No data found for customer_id={customer_id}.")
    st.stop()

X = features[["transaction_count", "total_spent"]]
y_prob = model.predict_proba(X)[0][1]  # probability of churn

# Display results
st.subheader(f"Prediction for Customer ID {customer_id}")
st.write(f"Churn Probability: **{y_prob:.2%}**")

# Gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=y_prob * 100,
    title={"text": "Churn Risk (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "red" if y_prob > 0.5 else "green"},
        "steps": [
            {"range": [0, 50], "color": "lightgreen"},
            {"range": [50, 75], "color": "yellow"},
            {"range": [75, 100], "color": "red"},
        ],
    }
))
st.plotly_chart(fig, use_container_width=True)