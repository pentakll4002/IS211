import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def run_kmeans(rfm_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = ['frequency', 'recency', 'monetary']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_raw')
    rfm_assembled = assembler.transform(rfm_df)

    scaler = StandardScaler(inputCol='features_raw', outputCol='features',
                            withMean=True, withStd=True)
    rfm_scaled = scaler.fit(rfm_assembled).transform(rfm_assembled)

    evaluator = ClusteringEvaluator(predictionCol='cluster', featuresCol='features')
    k_range = range(2, 8)
    silhouette_scores = []
    wssse_scores = []

    for k in k_range:
        km = KMeans(k=k, seed=42, featuresCol='features', predictionCol='cluster')
        model = km.fit(rfm_scaled)
        preds = model.transform(rfm_scaled)
        sil = evaluator.evaluate(preds)
        wssse = model.summary.trainingCost
        silhouette_scores.append(sil)
        wssse_scores.append(wssse)
        print(f'  k={k}: Silhouette={sil:.4f}, WSSSE={wssse:.2f}')

    best_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f'Best k = {best_k}')

    km_final = KMeans(k=best_k, seed=42, featuresCol='features', predictionCol='cluster')
    km_model = km_final.fit(rfm_scaled)
    clustered = km_model.transform(rfm_scaled)
    final_sil = evaluator.evaluate(clustered)
    print(f'Final Silhouette: {final_sil:.4f}')

    try:
        from sklearn.metrics import davies_bouldin_score
        from pyspark.ml.functions import vector_to_array
        eval_pdf = clustered.select(
            vector_to_array('features').alias('features'), 'cluster'
        ).toPandas()
        X_eval = np.array(eval_pdf['features'].tolist())
        db_score = davies_bouldin_score(X_eval, eval_pdf['cluster'].values)
        print(f'Davies-Bouldin Index: {db_score:.4f}')
    except Exception as e:
        print(f'Davies-Bouldin skipped: {e}')

    summary = clustered.groupBy('cluster').agg(
        F.count('user_id').alias('count'),
        F.mean('frequency').alias('avg_frequency'),
        F.mean('recency').alias('avg_recency'),
        F.mean('monetary').alias('avg_monetary')
    ).orderBy('cluster').toPandas()
    print(summary.to_string(index=False))

    _plot_elbow(k_range, wssse_scores, silhouette_scores, output_dir)
    _plot_clusters(clustered, best_k, final_sil, output_dir)

    return km_model, clustered, final_sil

def _plot_elbow(k_range, wssse, sil, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_range), wssse, 'bo-')
    axes[0].set_title('Elbow Method (WSSSE)')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('WSSSE')
    axes[1].plot(list(k_range), sil, 'ro-')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kmeans_elbow.png'), dpi=100)
    plt.close()

def _plot_clusters(clustered, best_k, sil, output_dir):
    pdf = clustered.select('frequency', 'monetary', 'cluster').toPandas()
    plt.figure(figsize=(10, 6))
    for c in sorted(pdf['cluster'].unique()):
        sub = pdf[pdf['cluster'] == c]
        plt.scatter(sub['frequency'], sub['monetary'], label=f'Cluster {c}', alpha=0.5, s=10)
    plt.xlabel('Frequency')
    plt.ylabel('Monetary')
    plt.title(f'K-Means (k={best_k}, Silhouette={sil:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kmeans_clusters.png'), dpi=100)
    plt.close()
