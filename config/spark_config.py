import os

HADOOP_HOME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hadoop')
os.environ['HADOOP_HOME'] = HADOOP_HOME
os.environ['PATH'] = os.path.join(HADOOP_HOME, 'bin') + ';' + os.environ.get('PATH', '')

from pyspark.sql import SparkSession

APP_NAME = 'SupermarketAnalytics'
MASTER = 'local[*]'
EXECUTOR_MEMORY = '4g'
DRIVER_MEMORY = '4g'
SHUFFLE_PARTITIONS = '8'
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(DATA_DIR, 'ECommerce_consumer behaviour.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'spark_output')

def get_spark(app_suffix=''):
    name = f'{APP_NAME}_{app_suffix}' if app_suffix else APP_NAME
    spark = SparkSession.builder \
        .appName(name) \
        .master(MASTER) \
        .config('spark.executor.memory', EXECUTOR_MEMORY) \
        .config('spark.driver.memory', DRIVER_MEMORY) \
        .config('spark.sql.shuffle.partitions', SHUFFLE_PARTITIONS) \
        .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    return spark
