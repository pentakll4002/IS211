import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.spark_config import get_spark, OUTPUT_DIR
from modules.data_loader import load_parquet
from modules.feature_engineer import build_rfm
from modules.clustering import run_kmeans
from modules.recommender import run_als
from modules.forecasting import run_arima

def main():
    spark = get_spark('Pipeline02_Training')
    print(f'Spark {spark.version} | {spark.sparkContext.master}')
    print(f'UI: {spark.sparkContext.uiWebUrl}')

    df = load_parquet(spark, os.path.join(OUTPUT_DIR, 'cleaned_data.parquet'))

    print('\n' + '=' * 60)
    print('1. K-MEANS CLUSTERING (Distributed Spark)')
    print('=' * 60)
    rfm = build_rfm(df)
    km_model, clustered, sil = run_kmeans(rfm, OUTPUT_DIR)

    print('\n' + '=' * 60)
    print('2. ALS COLLABORATIVE FILTERING (Distributed Spark)')
    print('=' * 60)
    als_model, rmse_als = run_als(df)

    print('\n' + '=' * 60)
    print('3. ARIMA TIME SERIES FORECASTING')
    print('=' * 60)
    rmse_arima = run_arima(df, OUTPUT_DIR)

    print('\n' + '=' * 60)
    print('4. KNN ITEM-BASED COLLABORATIVE FILTERING')
    print('=' * 60)
    pdf = df.toPandas()
    from modules.knn_recommender import run_knn_recommender
    knn_results = run_knn_recommender(pdf, OUTPUT_DIR)

    print('\n' + '=' * 60)
    print('5. NEURAL COLLABORATIVE FILTERING (PyTorch NCF)')
    print('=' * 60)
    from modules.ncf_model import run_ncf
    ncf_results = run_ncf(pdf, OUTPUT_DIR)

    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    print(f'K-Means Silhouette: {sil:.4f}')
    print(f'ALS RMSE: {rmse_als:.4f}')
    print(f'ARIMA RMSE: {rmse_arima:.4f}')
    print(f'KNN CF - Precision: {knn_results["precision"]:.4f}, Recall: {knn_results["recall"]:.4f}, F1: {knn_results["f1"]:.4f}, AUC: {knn_results["auc"]:.4f}')
    print(f'NCF    - Precision: {ncf_results["precision"]:.4f}, Recall: {ncf_results["recall"]:.4f}, F1: {ncf_results["f1"]:.4f}, AUC: {ncf_results["auc"]:.4f}')
    print('=' * 60)

    print(f'\nSpark UI: {spark.sparkContext.uiWebUrl}')

    # Only prompt in local/interactive mode (skip in Docker/automated runs)
    mode = os.environ.get('SPARK_MODE', 'local').lower()
    if mode == 'local':
        input('Press Enter to stop Spark...')

    spark.stop()

if __name__ == '__main__':
    main()

