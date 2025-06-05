# MLOps Feature Store w/ Feast & Redis
In this project, we will build a simple yet practical MLOps project that demonstrates a local feature store with Parquet file offline feature store and Redis online feature store. We will also "materialize or load" feature values from the offline store into the Redis online store. Then, we will read the latest features from the Redis online store to serve features for the training of a Machine Learning model and inferencing.


## Set Up Environment 

### Tech Stack
1. Python:          Programming Language
2. Feast:           Feature Store Framework
3. Redis:           Online Feature Store and local file (parquet) for Offline Storage
4. Pandas:          Numerical Computations
5. Scikit-learn:    Machine Learning Python Package
6. Streamlit:       Package ML model with Interactive web GUI
7. Docker:          Package ML model & all dependencies in a single image file or container.
8. Plotly:          Generating a Gauge chart to display probability

### Set up Project Environment

```sh
python3 -m venv venv
source venv/bin/activate
```

### Install Required Packages
```sh
pip install -r requirements.txt
```

### Start Redis with Docker
To ensure effective communication between the Redis & RedisInsight containers (LATER), they must be put on the same network: 

```sh
docker network ls
```
```sh
docker network create redis-network
```

```sh
docker run -d --name redis --network redis-network redis
```
```sh
docker ps
```

To get the IP address of Redis Docker Container: 
```sh
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' redis
```
Copy IP Address and Save it somewhere safe.


### Initialize Feast Repository
```sh
feast init my_feature_repo
cd my_feature_repo
mkdir data
```

### Configure Feast to Use Redis & File Store
In the my_feature_repo/feature_repo/feature_store.yaml file, delete content & add the ff:

```sh
project: my_feature_repo
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: "redisIP:6379"
offline_store:
  type: file
```


### Create Dataset
Create the file my_feature_repo/generate_data.py and add the ff: 
Make sure you created the data directory earlier

```sh
import pandas as pd
from datetime import datetime, timedelta

def generate_customer_data():
    customers = []
    start_date = datetime.now() - timedelta(days=30)
    
    for customer_id in range(1, 6):
        for day in range(30):
            date = start_date + timedelta(days=day)
            customers.append({
                "customer_id": customer_id,
                "transaction_count": day + customer_id,
                "total_spent": (day + 1) * 5.0,
                "event_timestamp": date
            })

    df = pd.DataFrame(customers)
    df.to_csv("data/customer_transactions.csv", index=False)
    df.to_parquet("data/customer_transactions.parquet")

if __name__ == "__main__":
    generate_customer_data()
```


### Run the generate_data.py file
```sh
python generate_data.py
```
 

### Define Features with Feast
Create the file feature_repo/example_feature_repo.py and add the ff contents: 

```sh
from datetime import timedelta
from feast import Entity, Field, FeatureView, FileSource, ValueType
from feast.types import Int64, Float32

customer_source = FileSource(
    path="../data/customer_transactions.parquet",
    event_timestamp_column="event_timestamp"
)

customer = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="Customer ID"
)

customer_fv = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="transaction_count", dtype=Int64),
        Field(name="total_spent", dtype=Float32),
    ],
    source=customer_source
)
```


### Apply Feature Definitions

```sh
cd feature_repo
feast apply
```

You can ignore the long list of "protobuf" warnings using the command:
```sh
PYTHONWARNINGS="ignore" feast apply
```
If it works, then you've successfully registered your features definitions.


### Materialize Data to Online Store (Redis)
This loads the most recent features into the online store (Redis) so they can be queried in real time.

```sh
cd feature_repo
feast materialize-incremental $(date +%F)
```
You should see logs like below: 
```sh
Materializing 3 feature views to 2025-05-23 00:00:00+00:00 into the redis online store.
```


### Get a Simple Web UI for Redis with Redis Insight Docker Container
To ensure effective communication between the Redis & RedisInsight containers, they must be put on the same network: 

```sh
docker run -d --name redisinsight --network redis-network -p 5540:5540 redis/redisinsight:latest
```

Open redinsight GUI on your browser: 
```sh
localhost:5540
```

Note: If you want data to persist, attach a volume to the RedisInsight container: 
```sh
docker run -d --name redisinsight --network redis-network -p 5540:5540 redis/redisinsight:latest -v redisinsight:/data
```

Add Redis DB in the Redis Insight GUI > Connection Settings:
```sh
    Alias: some_alias
    Host: redisIP
    Port: 6379
```
Test Connection > Connection should be successful!


### Retrieve Features from Redis Online Feature Store for Inference
Create and use the file get_online_features.py to get features stored in the online DB (a.k.a Redis) with the ff content:

```sh
cd feature_repo
```

```sh
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
``` 

Run the python script:
```sh
python get_online_features.py
```
You should get output on terminal. Inspect the Redis Insight GUI as well (Use protobuf)


### Train Machine Learning Model
Create the ML model training script with content below and run: 

```sh
cd my_feature_repo
touch train_model.py
```

```sh
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
```

```sh
python train_model.py
```


### Create & Run Streamlit App 
```sh
cd my_feature_repo
touch streamlit-app.py
```

```sh
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
```

```sh
streamlit run streamlit-app.py
```

On your browser, open:
```sh
localhost:8501
```

### Package Trained ML Model with Docker
Make sure Docker is installed on your system & requirements.txt should be in the same directory as the Dockerfile (my_feature_repo):

```sh
cd my_feature_repo
touch Dockerfile
```

Add the following contents: 
```sh
# Use an official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy app and model
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "streamlit-app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```


Build the image:
```sh
docker build -t feast-streamlit-app .
```

### Run the Streamlit Container 
Make sure to connect the Streamlit container to the Redis Container via the Docker network

```sh
docker run -d -p 8501:8501 --network redis-network --name streamlit-app feast-streamlit-app
```

On your browser, open:
```sh
localhost:8501
```
## Clean up
Stop all the running Docker Containers
```sh
docker stop <containerID>
```

Remove the stopped Docker Containers
```sh
docker remove <containerID>
```

Remove the docker image (if you don't need it anymore)
```sh
docker rmi <imageID>
```

Deactivate and Remove the Virtual Environment
```sh
cd FeatureStore_with_Feast_Redis
deactivate
rm -rf venv
```

**Please Like, Comment, and Subscribe to iQuant on YouTube**
