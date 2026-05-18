from pyspark.sql import functions as F

def build_user_features(df):
    user_features = df.groupBy('user_id').agg(
        F.countDistinct('order_id').alias('total_orders'),
        F.count('product_id').alias('total_items_purchased'),
        F.mean('days_since_prior_order').alias('avg_days_between_orders'),
        F.mean('reordered').alias('avg_reorder_rate'),
        F.countDistinct('department').alias('unique_departments')
    )
    print(f'User features: {user_features.count():,} users')
    return user_features

def build_rfm(df):
    rfm = df.groupBy('user_id').agg(
        F.max('order_number').alias('frequency'),
        F.mean(F.when(F.col('days_since_prior_order') >= 0,
                      F.col('days_since_prior_order'))).alias('recency'),
        F.count('product_id').alias('monetary')
    ).na.fill(0)

    for col_name in ['frequency', 'recency', 'monetary']:
        stats = rfm.agg(
            F.mean(col_name).alias('mean'),
            F.stddev(col_name).alias('std')
        ).collect()[0]
        rfm = rfm.withColumn(
            f'{col_name}_zscore',
            (F.col(col_name) - stats['mean']) / stats['std']
        )

    n_before = rfm.count()
    rfm_clean = rfm.filter(
        (F.abs(F.col('frequency_zscore')) < 3) &
        (F.abs(F.col('recency_zscore')) < 3) &
        (F.abs(F.col('monetary_zscore')) < 3)
    )
    n_after = rfm_clean.count()
    print(f'RFM: {n_before:,} -> {n_after:,} (removed {n_before - n_after:,} outliers)')
    return rfm_clean

def build_order_features(df):
    order_features = df.groupBy('order_id').agg(
        F.first('user_id').alias('user_id'),
        F.first('order_dow').alias('order_dow'),
        F.first('order_hour_of_day').alias('order_hour_of_day'),
        F.count('product_id').alias('basket_size'),
        F.mean('reordered').alias('reorder_ratio'),
        F.countDistinct('department').alias('department_count')
    )
    print(f'Order features: {order_features.count():,} orders')
    return order_features
