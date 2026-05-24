import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score

def label_transfer(ref, query, embed="X_emb", label_key="celltype", k=15, metric="cosine"):
    X_train = np.asarray(ref.obsm[embed]) if embed in ref.obsm else np.asarray(ref.X)
    X_test = np.asarray(query.obsm[embed]) if embed in query.obsm else np.asarray(query.X)
    y_train = np.asarray(ref.obs[label_key])
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)


def calculate_shared_type_transfer_metrics(adata, batch_key, label_key, embed="X_emb"):
    batches = adata.obs[batch_key].unique()
    if len(batches) != 2:
        raise ValueError("label_transfer currently supports exactly two batches.")
    adata_A = adata[adata.obs[batch_key] == batches[0]]
    adata_B = adata[adata.obs[batch_key] == batches[1]]
    y_A, y_B = adata_A.obs[label_key], adata_B.obs[label_key]
    pred_A = label_transfer(adata_B, adata_A, embed=embed, label_key=label_key)
    pred_B = label_transfer(adata_A, adata_B, embed=embed, label_key=label_key)
    
    cell_types_A = set(y_A.unique())
    cell_types_B = set(y_B.unique())
    shared_cell_types = cell_types_A.intersection(cell_types_B)
    if not shared_cell_types:
        raise ValueError("两个批次之间没有共享的细胞类型，无法计算共享类型的转移准确率。")
    
    mask_A = y_A.isin(shared_cell_types)
    y_A_shared = y_A[mask_A]
    pred_A_shared = pred_A[mask_A] 
    
    mask_B = y_B.isin(shared_cell_types)
    y_B_shared = y_B[mask_B]
    pred_B_shared = pred_B[mask_B]
    
    transfer_acc_shared = (accuracy_score(y_A_shared, pred_A_shared) + accuracy_score(y_B_shared, pred_B_shared)) / 2
    
    transfer_f1_shared = (f1_score(y_A_shared, pred_A_shared, average='macro') + 
                          f1_score(y_B_shared, pred_B_shared, average='macro')) / 2
    
    return transfer_acc_shared, transfer_f1_shared


def mean_average_precision(adata, embed="X_emb", label_key="celltype", neighbor_frac=0.01, **kwargs):
    x = np.asarray(adata.obsm[embed]) if embed in adata.obsm and adata.obsm[embed] is not None else np.asarray(adata.X)
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' not found in adata.obs.")
    y = np.asarray(adata.obs[label_key])
    n_samples = y.shape[0]
    k = max(round(n_samples * neighbor_frac), 1)
    nn = NearestNeighbors(n_neighbors=min(n_samples, k + 1), **kwargs).fit(x)
    nni = nn.kneighbors(x, return_distance=False)[:, 1:]
    match = np.equal(y[nni], np.expand_dims(y, 1))

    def _average_precision(row):
        if np.any(row):
            cummean = np.cumsum(row) / (np.arange(len(row)) + 1)
            return cummean[row].mean()
        return 0.0

    map_score = np.apply_along_axis(_average_precision, 1, match).mean()
    return float(map_score)



