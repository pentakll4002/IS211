from pyspark.sql import functions as F

def clean_data(df):
    before = df.count()

    median_days = df.approxQuantile('days_since_prior_order', [0.5], 0.01)[0]
    df = df.fillna({'days_since_prior_order': median_days})
    df = df.fillna({'department': 'unknown', 'product_name': 'unknown'})
    df = df.dropna(subset=['order_id', 'user_id', 'product_id'])

    for col_name in ['order_id', 'user_id', 'product_id', 'department_id']:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast('int'))

    after = df.count()
    print(f'Cleaning: {before:,} -> {after:,} rows (removed {before - after:,})')
    return df
