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