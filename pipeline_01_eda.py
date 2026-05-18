import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.spark_config import get_spark, CSV_PATH, OUTPUT_DIR
from modules.data_loader import load_csv, save_parquet
from modules.data_cleaner import clean_data
from modules.feature_engineer import build_user_features, build_order_features
from modules.eda import run_eda
from modules.association_rules import run_fpgrowth

def main():
    spark = get_spark('Pipeline01_EDA')
    print(f'Spark {spark.version} | {spark.sparkContext.master}')
    print(f'UI: {spark.sparkContext.uiWebUrl}')

    df = load_csv(spark, CSV_PATH)
    df = clean_data(df)

    user_features = build_user_features(df)
    order_features = build_order_features(df)

    run_eda(df, OUTPUT_DIR)

    run_fpgrowth(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_parquet(df, os.path.join(OUTPUT_DIR, 'cleaned_data.parquet'))
    save_parquet(user_features, os.path.join(OUTPUT_DIR, 'user_features.parquet'))
    save_parquet(order_features, os.path.join(OUTPUT_DIR, 'order_features.parquet'))

    print('Pipeline 01 completed')
    spark.stop()

if __name__ == '__main__':
    main()
