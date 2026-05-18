from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def run_als(df, sample_users=2000, sample_products=200):
    top_users = df.groupBy('user_id').count() \
        .orderBy(F.desc('count')).limit(sample_users).select('user_id')
    top_products = df.groupBy('product_id').count() \
        .orderBy(F.desc('count')).limit(sample_products).select('product_id')

    ratings = df.join(top_users, 'user_id', 'inner') \
        .join(top_products, 'product_id', 'inner') \
        .groupBy('user_id', 'product_id') \
        .agg(F.count('*').alias('rating'))
    ratings = ratings.withColumn('user_id', F.col('user_id').cast('int')) \
        .withColumn('product_id', F.col('product_id').cast('int')) \
        .withColumn('rating', F.col('rating').cast('float'))

    print(f'Ratings matrix: {ratings.count():,}')

    train, test = ratings.randomSplit([0.8, 0.2], seed=42)
    print(f'Train: {train.count():,} | Test: {test.count():,}')

    als = ALS(
        rank=10, maxIter=10, regParam=0.1,
        userCol='user_id', itemCol='product_id', ratingCol='rating',
        coldStartStrategy='drop', implicitPrefs=True
    )
    model = als.fit(train)
    print('ALS model trained')

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName='rmse', labelCol='rating', predictionCol='prediction'
    )
    rmse = evaluator.evaluate(predictions)
    print(f'RMSE: {rmse:.4f}')

    top_recs = model.recommendForAllUsers(5)
    print('Top recommendations:')
    top_recs.show(3, truncate=False)

    return model, rmse
