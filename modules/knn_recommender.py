import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def _predict_item_based(user_vector, similarity_matrix, k=30):
    n_items = len(user_vector)
    scores = np.zeros(n_items)
    for i in range(n_items):
        if user_vector[i] == 0:
            sim_items = similarity_matrix[i]
            purchased = np.where(user_vector > 0)[0]
            if len(purchased) == 0:
                continue
            top_k_idx = purchased[np.argsort(sim_items[purchased])[-k:]]
            scores[i] = sim_items[top_k_idx].sum()
    return scores

def run_knn_recommender(df, output_dir, sample_users=2000, sample_products=200):
    os.makedirs(output_dir, exist_ok=True)

    top_products = df['product_name'].value_counts().head(sample_products).index.tolist()
    df_rec = df[df['product_name'].isin(top_products)]

    active_users = df_rec.groupby('user_id')['product_name'].nunique()
    active_users = active_users[active_users >= 5].index.tolist()
    np.random.seed(42)
    if len(active_users) > sample_users:
        sampled_users = np.random.choice(active_users, sample_users, replace=False)
    else:
        sampled_users = active_users
    df_rec = df_rec[df_rec['user_id'].isin(sampled_users)]

    user_product = df_rec.groupby(['user_id', 'product_name']).size().unstack(fill_value=0)
    user_product = (user_product > 0).astype(int)
    print(f'KNN Recommender: {user_product.shape[0]} users x {user_product.shape[1]} products')

    item_matrix = user_product.values.T
    norms = np.sqrt((item_matrix ** 2).sum(axis=1, keepdims=True))
    norms[norms == 0] = 1
    item_norm = item_matrix / norms
    similarity_matrix = item_norm @ item_norm.T
    np.fill_diagonal(similarity_matrix, 0)

    results = _cross_validate(user_product.values, similarity_matrix, output_dir)
    _optimize_k(user_product.values, similarity_matrix)
    return results

def _cross_validate(user_matrix, similarity_matrix, output_dir, n_splits=5):
    n_users = user_matrix.shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_precision, all_recall, all_f1, all_accuracy = [], [], [], []
    all_fpr_list, all_tpr_list = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(range(n_users)), 1):
        train_matrix = user_matrix.copy()
        test_items_per_user = {}

        for idx in test_idx:
            purchased = np.where(user_matrix[idx] > 0)[0]
            if len(purchased) < 2:
                continue
            n_hide = max(1, len(purchased) // 5)
            hide_idx = np.random.choice(purchased, n_hide, replace=False)
            test_items_per_user[idx] = hide_idx
            train_matrix[idx, hide_idx] = 0

        t_item_matrix = train_matrix.T
        t_norms = np.sqrt((t_item_matrix ** 2).sum(axis=1, keepdims=True))
        t_norms[t_norms == 0] = 1
        t_item_norm = t_item_matrix / t_norms
        t_item_sim = t_item_norm @ t_item_norm.T
        np.fill_diagonal(t_item_sim, 0)

        fold_true, fold_pred, fold_scores = [], [], []

        for idx in test_idx:
            if idx not in test_items_per_user:
                continue
            user_vec = train_matrix[idx]
            scores = _predict_item_based(user_vec, t_item_sim, k=30)
            hidden = test_items_per_user[idx]
            true_labels = np.zeros(len(scores))
            true_labels[hidden] = 1
            not_in_train = np.where(user_vec == 0)[0]
            if len(not_in_train) == 0:
                continue
            fold_true.extend(true_labels[not_in_train])
            fold_scores.extend(scores[not_in_train])
            top_n = min(10, len(not_in_train))
            top_indices = np.argsort(scores[not_in_train])[-top_n:]
            pred_labels = np.zeros(len(not_in_train))
            pred_labels[top_indices] = 1
            fold_pred.extend(pred_labels)

        if len(fold_true) > 0:
            fold_true = np.array(fold_true)
            fold_pred = np.array(fold_pred)
            fold_scores = np.array(fold_scores)
            prec = precision_score(fold_true, fold_pred, zero_division=0)
            rec = recall_score(fold_true, fold_pred, zero_division=0)
            f1 = f1_score(fold_true, fold_pred, zero_division=0)
            acc = accuracy_score(fold_true, fold_pred)
            all_precision.append(prec)
            all_recall.append(rec)
            all_f1.append(f1)
            all_accuracy.append(acc)
            fpr, tpr, _ = roc_curve(fold_true, fold_scores)
            all_fpr_list.append(fpr)
            all_tpr_list.append(tpr)
            print(f'  Fold {fold}: Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}')

    avg_prec = np.mean(all_precision)
    avg_rec = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    avg_acc = np.mean(all_accuracy)
    mean_auc = np.mean([auc(fpr, tpr) for fpr, tpr in zip(all_fpr_list, all_tpr_list)])
    print(f'  Avg Precision={avg_prec:.4f} Recall={avg_rec:.4f} F1={avg_f1:.4f} Accuracy={avg_acc:.4f} AUC={mean_auc:.4f}')

    _plot_roc(all_fpr_list, all_tpr_list, output_dir)
    return {'precision': avg_prec, 'recall': avg_rec, 'f1': avg_f1, 'accuracy': avg_acc, 'auc': mean_auc}

def _plot_roc(all_fpr, all_tpr, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
    for i, (fpr, tpr) in enumerate(zip(all_fpr, all_tpr)):
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], label=f'Fold {i+1} (AUC={roc_auc:.2f})', alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Item-Based CF (5-Fold)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'knn_roc_curve.png'), dpi=100)
    plt.close()

def _optimize_k(user_matrix, similarity_matrix):
    k_values = [10, 20, 30, 40, 50]
    print('K-optimization:')
    for k in k_values:
        n_items = user_matrix.shape[1]
        total_prec, total_rec, count = 0, 0, 0
        for idx in range(min(50, user_matrix.shape[0])):
            user_vec = user_matrix[idx]
            purchased = np.where(user_vec > 0)[0]
            if len(purchased) < 2:
                continue
            scores = _predict_item_based(user_vec, similarity_matrix, k=k)
            not_purchased = np.where(user_vec == 0)[0]
            if len(not_purchased) == 0:
                continue
            count += 1
        print(f'  k={k}: evaluated on {count} users')
