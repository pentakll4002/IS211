import os

# ============================================================
# Spark Config — Multi-mode: local | distributed | docker
# Set environment variable SPARK_MODE to switch:
#   local        → single JVM, chạy nhanh (default)
#   distributed  → Standalone cluster trên Windows
#   docker       → Docker Compose cluster
# ============================================================

MODE = os.environ.get('SPARK_MODE', 'local').lower()

# Hadoop setup — only needed on Windows (local/distributed)
if MODE != 'docker':
    HADOOP_HOME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hadoop')
    os.environ['HADOOP_HOME'] = HADOOP_HOME
    path_sep = ';' if os.name == 'nt' else ':'
    os.environ['PATH'] = os.path.join(HADOOP_HOME, 'bin') + path_sep + os.environ.get('PATH', '')

from pyspark.sql import SparkSession

APP_NAME = 'SupermarketAnalytics'

# Mode-specific configuration
if MODE == 'docker':
    MASTER = 'spark://spark-master:7077'
    EXECUTOR_MEMORY = '2g'
    DRIVER_MEMORY = '2g'
    SHUFFLE_PARTITIONS = '4'
    DATA_DIR = '/app'
elif MODE == 'distributed':
    MASTER = 'spark://localhost:7077'
    EXECUTOR_MEMORY = '2g'
    DRIVER_MEMORY = '2g'
    SHUFFLE_PARTITIONS = '4'
    DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:  # local (default)
    MASTER = 'local[*]'
    EXECUTOR_MEMORY = '4g'
    DRIVER_MEMORY = '4g'
    SHUFFLE_PARTITIONS = '8'
    DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(DATA_DIR, 'ECommerce_consumer behaviour.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'spark_output')

def get_spark(app_suffix=''):
    name = f'{APP_NAME}_{app_suffix}' if app_suffix else APP_NAME
    builder = SparkSession.builder \
        .appName(name) \
        .master(MASTER) \
        .config('spark.executor.memory', EXECUTOR_MEMORY) \
        .config('spark.driver.memory', DRIVER_MEMORY) \
        .config('spark.sql.shuffle.partitions', SHUFFLE_PARTITIONS)

    # Extra configs for cluster modes
    if MODE in ('distributed', 'docker'):
        builder = builder \
            .config('spark.executor.cores', '2') \
            .config('spark.cores.max', '4') \
            .config('spark.default.parallelism', '4') \
            .config('spark.sql.adaptive.enabled', 'true')

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print(f'[Config] Mode={MODE} | Master={MASTER}')
    return spark
