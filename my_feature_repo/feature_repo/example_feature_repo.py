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