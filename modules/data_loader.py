from pyspark.sql import functions as F

def load_csv(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f'Loaded {df.count():,} rows, {len(df.columns)} columns')
    df.printSchema()
    return df

def load_parquet(spark, path):
    df = spark.read.parquet(path)
    print(f'Loaded parquet: {df.count():,} rows')
    return df

def save_parquet(df, path):
    df.write.mode('overwrite').parquet(path)
    print(f'Saved: {path}')
