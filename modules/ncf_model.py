import numpy as np
import os

def run_ncf(df, output_dir, sample_users=2000, sample_products=200, epochs=10, batch_size=512, embed_dim=32):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'NCF device: {device}')

    ncf_df = df[['user_id', 'product_name']].drop_duplicates().copy()

    top_products = df['product_name'].value_counts().head(sample_products).index.tolist()
    active_users = df[df['product_name'].isin(top_products)].groupby('user_id')['product_name'].nunique()
    active_users = active_users[active_users >= 3].index.tolist()
    np.random.seed(42)
    if len(active_users) > sample_users:
        active_users = list(np.random.choice(active_users, sample_users, replace=False))

    ncf_df = ncf_df[(ncf_df['user_id'].isin(active_users)) & (ncf_df['product_name'].isin(top_products))]

    user_ids = ncf_df['user_id'].unique()
    product_names = ncf_df['product_name'].unique()
    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    product2idx = {pname: i for i, pname in enumerate(product_names)}
    n_users = len(user2idx)
    n_products = len(product2idx)
    ncf_df['user_idx'] = ncf_df['user_id'].map(user2idx)
    ncf_df['product_idx'] = ncf_df['product_name'].map(product2idx)
    print(f'NCF: {n_users} users, {n_products} products, {len(ncf_df)} interactions')

    positive_set = set(zip(ncf_df['user_idx'].values, ncf_df['product_idx'].values))
    neg_users, neg_products = [], []
    neg_ratio = 4
    for u_idx, p_idx in positive_set:
        count = 0
        while count < neg_ratio:
            neg_p = np.random.randint(0, n_products)
            if (u_idx, neg_p) not in positive_set:
                neg_users.append(u_idx)
                neg_products.append(neg_p)
                count += 1

    all_users = np.concatenate([ncf_df['user_idx'].values, np.array(neg_users)])
    all_products = np.concatenate([ncf_df['product_idx'].values, np.array(neg_products)])
    all_labels = np.concatenate([np.ones(len(ncf_df)), np.zeros(len(neg_users))])
    print(f'Total samples: {len(all_labels)} (Pos: {int(all_labels.sum())}, Neg: {int(len(all_labels) - all_labels.sum())})')

    indices = np.arange(len(all_labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=all_labels)

    class NCFDataset(Dataset):
        def __init__(self, users, products, labels):
            self.users = torch.LongTensor(users)
            self.products = torch.LongTensor(products)
            self.labels = torch.FloatTensor(labels)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.users[idx], self.products[idx], self.labels[idx]

    class NeuralCF(nn.Module):
        def __init__(self, n_users, n_products, embed_dim=32, mlp_dims=[64, 32, 16]):
            super().__init__()
            self.gmf_user = nn.Embedding(n_users, embed_dim)
            self.gmf_product = nn.Embedding(n_products, embed_dim)
            self.mlp_user = nn.Embedding(n_users, embed_dim)
            self.mlp_product = nn.Embedding(n_products, embed_dim)
            layers = []
            in_dim = embed_dim * 2
            for dim in mlp_dims:
                layers.extend([nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(0.2)])
                in_dim = dim
            self.mlp = nn.Sequential(*layers)
            self.output = nn.Linear(embed_dim + mlp_dims[-1], 1)
            self.sigmoid = nn.Sigmoid()
            self._init()
        def _init(self):
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.01)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        def forward(self, u, p):
            gmf = self.gmf_user(u) * self.gmf_product(p)
            mlp_in = torch.cat([self.mlp_user(u), self.mlp_product(p)], dim=-1)
            mlp_out = self.mlp(mlp_in)
            out = self.sigmoid(self.output(torch.cat([gmf, mlp_out], dim=-1)))
            return out.squeeze()

    train_ds = NCFDataset(all_users[train_idx], all_products[train_idx], all_labels[train_idx])
    test_ds = NCFDataset(all_users[test_idx], all_products[test_idx], all_labels[test_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = NeuralCF(n_users, n_products, embed_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'NCF params: {total_params:,}')

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, p, l in train_dl:
            u, p, l = u.to(device), p.to(device), l.to(device)
            optimizer.zero_grad()
            pred = model(u, p)
            loss = criterion(pred, l)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}')

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for u, p, l in test_dl:
            u, p = u.to(device), p.to(device)
            pred = model(u, p).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(l.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    binary_preds = (all_preds >= 0.5).astype(int)
    prec = precision_score(all_true, binary_preds, zero_division=0)
    rec = recall_score(all_true, binary_preds, zero_division=0)
    f1 = f1_score(all_true, binary_preds, zero_division=0)
    try:
        auc_score = roc_auc_score(all_true, all_preds)
    except:
        auc_score = 0.0
    print(f'NCF Results: Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} AUC={auc_score:.4f}')

    return {'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_score, 'model': model}
