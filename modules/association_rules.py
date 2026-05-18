from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth

def run_fpgrowth(df, min_support=0.005, min_confidence=0.1, sample_frac=0.05):
    sampled = df.sample(False, sample_frac, seed=42)
    print(f'FP-Growth on {sampled.count():,} rows (sample={sample_frac})')

    basket_df = sampled.groupBy('order_id').agg(
        F.collect_set('product_name').alias('items')
    )
    basket_df = basket_df.filter(F.size('items') >= 2)

    fpGrowth = FPGrowth(
        itemsCol='items',
        minSupport=min_support,
        minConfidence=min_confidence
    )
    model = fpGrowth.fit(basket_df)

    freq_items = model.freqItemsets
    print(f'Frequent itemsets: {freq_items.count():,}')
    freq_items.orderBy(F.desc('freq')).show(10, truncate=50)

    rules = model.associationRules
    print(f'Association rules: {rules.count():,}')
    rules.orderBy(F.desc('confidence')).show(10, truncate=50)

    return model, freq_items, rules
