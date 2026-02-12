import warnings
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from numpy.linalg import norm
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import issparse, csc_matrix, csr_matrix
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

import anndata
import sklearn
import scanpy as sc

from collections import Counter
from typing import Optional, Callable, Union, Iterable

def lsi(adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var

    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)

    if issparse(X):
        X_norm = sklearn.preprocessing.normalize(X, norm="l1")
    else:
        X_norm = sklearn.preprocessing.normalize(X, norm="l1", axis=1)

    X_norm = np.log1p(X_norm * 1e4)

    U, S, _ = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)
    X_lsi = U * S

    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)

    adata.obsm["X_lsi"] = X_lsi


def tfidf(X: np.ndarray) -> np.ndarray:
    if issparse(X):
        row_sums = np.array(X.sum(axis=1)).flatten() + 1e-9
        tf = X.multiply(1 / row_sums[:, np.newaxis])
        idf = np.array(X.shape[0] / X.sum(axis=0)).flatten()
        return tf.multiply(idf)
    else:
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-9)
        idf = X.shape[0] / (X.sum(axis=0) + 1e-9)
        return tf * idf


def clr(adata: AnnData, axis: int = 0) -> None:
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")

    if issparse(adata.X) and axis == 0 and not isinstance(adata.X, csc_matrix):
        warnings.warn("adata.X is sparse but not in CSC format. Converting to CSC.")
        x = csc_matrix(adata.X)
    elif issparse(adata.X) and axis == 1 and not isinstance(adata.X, csr_matrix):
        warnings.warn("adata.X is sparse but not in CSR format. Converting to CSR.")
        x = csr_matrix(adata.X)
    else:
        x = adata.X

    if issparse(x):
        x.data /= np.repeat(
            np.exp(np.log1p(x).sum(axis=axis).A / x.shape[axis]),
            x.getnnz(axis=axis)
        )
        np.log1p(x.data, out=x.data)
    else:
        np.log1p(
            x / np.exp(np.log1p(x).sum(axis=axis, keepdims=True) / x.shape[axis]),
            out=x,
        )

    adata.X = x


def batch_scale(adata: AnnData, method: str = 'maxabs') -> None:
    if 'batch' in adata.obs:
        batches = adata.obs['batch'].unique()
    else:
        print("No 'batch' found in adata.obs, applying scaling to all data.")
        batches = [None]

    for b in batches:
        if b is None:
            idx = np.arange(adata.n_obs)
        else:
            idx = np.where(adata.obs['batch'] == b)[0]

        X_batch = adata.X[idx]

        if issparse(X_batch):
            if method == 'standard':
                scaler = StandardScaler(with_mean=False, copy=False).fit(X_batch)
                adata.X[idx] = scaler.transform(X_batch)
            elif method == 'maxabs':
                scaler = MaxAbsScaler(copy=False).fit(X_batch)
                adata.X[idx] = scaler.transform(X_batch)
            else:
                raise ValueError(f"Unknown scaling method: {method}. Choose 'maxabs' or 'standard'.")
        else:
            if method == 'standard':
                scaler = StandardScaler(copy=False).fit(X_batch)
            elif method == 'maxabs':
                scaler = MaxAbsScaler(copy=False).fit(X_batch)
            else:
                raise ValueError(f"Unknown scaling method: {method}. Choose 'maxabs' or 'standard'.")
            adata.X[idx] = scaler.transform(X_batch)


def build_celltype_prior(
    list1: List[Union[str, None]],
    list2: List[Union[str, None]]
) -> torch.Tensor:
    arr1 = np.array(list1, dtype=object)
    arr2 = np.array(list2, dtype=object)

    def missing_mask(arr: np.ndarray) -> np.ndarray:
        arr_str = np.char.lower(np.array(arr, str))
        str_missing = np.isin(arr_str, ["none", "nan", "na", "null", ""])
        none_missing = np.equal(arr, None)
        try:
            nan_missing = np.isnan(arr.astype(float))
        except Exception:
            nan_missing = np.zeros_like(arr, dtype=bool)
        return str_missing | none_missing | nan_missing

    mask1 = missing_mask(arr1)
    mask2 = missing_mask(arr2)

    eq_matrix = np.equal.outer(arr1, arr2).astype(np.float32)

    if mask1.any():
        eq_matrix[mask1, :] = 0
    if mask2.any():
        eq_matrix[:, mask2] = 0

    return torch.from_numpy(eq_matrix)


def pairwise_correlation_distance_sparse(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    topk: int = 5,
    high_cost: float = 1e8,    
    tau: float = 0.1
) -> torch.Tensor:

    Y = X if Y is None else Y
    ns, nt = X.size(0), Y.size(0)

    X_centered = X - X.mean(dim=1, keepdim=True)
    Y_centered = Y - Y.mean(dim=1, keepdim=True)
    cov = X_centered @ Y_centered.T
    std_X = torch.norm(X_centered, p=2, dim=1)
    std_Y = torch.norm(Y_centered, p=2, dim=1)
    corr = cov / (std_X.unsqueeze(1) * std_Y.unsqueeze(0) + 1e-8)
    C = 1 - corr 
    C = C / tau

    row_mask = torch.zeros_like(C, dtype=torch.bool)
    col_mask = torch.zeros_like(C, dtype=torch.bool)

    row_mask.scatter_(1, C.topk(k=min(topk, nt), dim=1, largest=False).indices, True)
    col_mask.scatter_(0, C.topk(k=min(topk, ns), dim=0, largest=False).indices, True)

    mnn_mask = row_mask & col_mask

    C_sparse = torch.ones_like(C) * high_cost
    C_sparse[mnn_mask] = C[mnn_mask]

    return C_sparse


def pairwise_correlation_distance(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    Y = X if Y is None else Y
    X_centered = X - X.mean(dim=1, keepdim=True)
    Y_centered = Y - Y.mean(dim=1, keepdim=True)
    cov = X_centered @ Y_centered.T
    std_X = torch.norm(X_centered, p=2, dim=1)
    std_Y = torch.norm(Y_centered, p=2, dim=1)
    corr = cov / (std_X.unsqueeze(1) * std_Y.unsqueeze(0) + 1e-8)
    return 1 - corr


def pairwise_euclidean_distance(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    clip: bool = False,
    clip_value: float = 1000.0
) -> torch.Tensor:
    Y = X if Y is None else Y
    if clip:
        max_norm = torch.max(
            torch.abs(X).max() + torch.abs(Y).max(),
            torch.tensor(2 * clip_value, device=X.device, dtype=X.dtype)
        ) / 2
        X, Y = clip_value * X / max_norm, clip_value * Y / max_norm
    X_col, Y_row = X.unsqueeze(1), Y.unsqueeze(0)
    return torch.mean((X_col - Y_row) ** 2, dim=-1)


def unbalanced_ot(
    cost_pp: torch.Tensor,
    reg: float = 0.05,
    reg_m: float = 0.5,
    prior: Optional[torch.Tensor] = None,
    device: str = 'cpu',
    max_iteration: Dict[str, int] = {'outer': 10, 'inner': 10},
) -> Optional[torch.Tensor]:
    ns, nt = cost_pp.shape
    if prior is not None:
        cost_pp = cost_pp * prior

    p_s = torch.ones(ns, 1, device=device) / ns
    p_t = torch.ones(nt, 1, device=device) / nt
    tran = torch.ones(ns, nt, device=device) / (ns * nt)
    dual = torch.ones(ns, 1, device=device) / ns
    f = reg_m / (reg_m + reg)

    for _ in range(max_iteration['outer']):
        cost = cost_pp
        kernel = torch.exp(-cost / (reg * torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for _ in range(max_iteration['inner']):
            dual = (p_s / (kernel @ b)) ** f
            b = (p_t / (torch.t(kernel) @ dual)) ** f
        tran = (dual @ torch.t(b)) * kernel

    out = tran.detach()

    return None if torch.isnan(out).sum() > 0 else out


def generalized_clip_loss_stable_masked(
    z_A: torch.Tensor,
    z_B: torch.Tensor,
    Y: torch.Tensor,
    tau: float = 0.1
) -> torch.Tensor:
    mask_A = Y.sum(dim=1) > 0
    mask_B = Y.sum(dim=0) > 0

    z_A_masked = z_A[mask_A]
    z_B_masked = z_B[mask_B]
    Y_masked = Y[mask_A][:, mask_B]

    z_A_masked = z_A_masked / z_A_masked.norm(dim=1, keepdim=True)
    z_B_masked = z_B_masked / z_B_masked.norm(dim=1, keepdim=True)
    S = z_A_masked @ z_B_masked.T / tau

    log_probs_A2B = F.log_softmax(S, dim=1)
    loss_A2B = -(Y_masked * log_probs_A2B).sum(dim=1) / (Y_masked.sum(dim=1) + 1e-8)
    loss_A2B = loss_A2B.mean()

    log_probs_B2A = F.log_softmax(S.T, dim=1)
    loss_B2A = -(Y_masked.T * log_probs_B2A).sum(dim=1) / (Y_masked.T.sum(dim=1) + 1e-8)
    loss_B2A = loss_B2A.mean()

    return (loss_A2B + loss_B2A) / 2


def log_nb_positive(x, mu, theta, eps=1e-8, return_sum=True):
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    if return_sum:
        res = torch.sum(res, dim=-1)
    return res


def zscore_numpy(X: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    X_z = (X - mean) / (std + eps)
    return X_z


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()

def build_mnn_prior(Sim: torch.Tensor, k: int) -> torch.Tensor:
    row_mask = torch.zeros_like(Sim, dtype=torch.bool)
    col_mask = torch.zeros_like(Sim, dtype=torch.bool)
    row_mask.scatter_(1, Sim.topk(k, dim=1, largest=False).indices, True)
    col_mask.scatter_(0, Sim.topk(k, dim=0, largest=False).indices, True)
    mnn_mask = row_mask & col_mask
    return torch.where(mnn_mask, 1.0, 0.0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))


def leiden_shared_mask(
    z_A: np.ndarray,
    z_B: np.ndarray,
    resolution: float = 1.0,
    min_shared_frac: float = 0.05,
    min_similarity: float = 0.8,
    random_state: int = 0,
    device: torch.device | None = None,
):
    n_A, n_B = z_A.shape[0], z_B.shape[0]

    Z = np.vstack([z_A, z_B])
    modality = np.array(["A"] * n_A + ["B"] * n_B)

    adata = anndata.AnnData(Z)
    adata.obs["modality"] = modality

    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        key_added="leiden",
    )

    clusters = adata.obs["leiden"].astype(int).values

    shared_cluster = {}
    for c in np.unique(clusters):
        idx = clusters == c
        mods = modality[idx]

        idx_A = idx & (modality == "A")
        idx_B = idx & (modality == "B")

        nA, nB = idx_A.sum(), idx_B.sum()
        total = nA + nB

        frac_ok = (
            min(nA, nB) / total >= min_shared_frac if total > 0 else False
        )

        sim_ok = False
        if nA > 0 and nB > 0:
            mu_A = z_A[idx_A[:n_A]].mean(axis=0)
            mu_B = z_B[idx_B[n_A:]].mean(axis=0)
            sim = cosine_sim(mu_A, mu_B)
            sim_ok = sim >= min_similarity

        shared_cluster[c] = frac_ok and sim_ok

    is_shared = np.array([shared_cluster[c] for c in clusters], dtype=bool)

    mask_A = is_shared[:n_A]
    mask_B = is_shared[n_A:]

    if device is not None:
        mask_A = torch.as_tensor(mask_A, dtype=torch.bool, device=device)
        mask_B = torch.as_tensor(mask_B, dtype=torch.bool, device=device)

    return mask_A, mask_B


def compute_celltype_weights(
    celltype_series_A: Union[pd.Series, Iterable],  
    celltype_series_B: Union[pd.Series, Iterable],  
    unique_labels: Union[np.ndarray, list],         
    weight_foo: Callable = np.sqrt,
    n_add: int = 0,
    device: Optional[torch.device] = None
) -> torch.Tensor:

    def make_class_weights(
        celltype_combined: Iterable,
        unique_labels: list,
        foo: Callable = np.sqrt,
        n_add: int = 0,
        astensor: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:

        counter = Counter(celltype_combined)
        counts = []
        for label in unique_labels:
            counts.append(counter.get(label, 0)) 
        counts = np.array(counts, dtype=np.int64)
        
        n_cls = len(unique_labels) + n_add
        weights = np.array([1 / foo(c + 1) if c > 0 else 0 for c in counts])
        
        if weights.sum() > 0:
            weights = weights / weights.sum() * (1 - n_add / n_cls)
        else:
            weights = np.zeros_like(weights)

        weights = np.concatenate([weights, np.array([1 / n_cls] * n_add)])
    
        if astensor:
            return torch.Tensor(weights.astype(np.float32))
        return weights

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if len(unique_labels) == 0:
        return torch.tensor([], device=device, dtype=torch.float32)
    
    celltype_list_A = celltype_series_A.tolist() if isinstance(celltype_series_A, pd.Series) else list(celltype_series_A)
    celltype_list_B = celltype_series_B.tolist() if isinstance(celltype_series_B, pd.Series) else list(celltype_series_B)
    celltype_combined = celltype_list_A + celltype_list_B
    
    weight = make_class_weights(
        celltype_combined=celltype_combined,
        unique_labels=unique_labels,
        foo=weight_foo,
        n_add=n_add,
        astensor=True
    )
    
    return weight.to(device, non_blocking=True)


def build_proto_and_update_dataset(
    z_A,
    z_B,
    dataset_A,
    dataset_B,
    num_classes: int,
    threshold: float = 0.7,
    normalize: bool = True,
    device: torch.device | None = None,
):

    if not torch.is_tensor(z_A):
        z_A = torch.tensor(z_A, dtype=torch.float32)
    if not torch.is_tensor(z_B):
        z_B = torch.tensor(z_B, dtype=torch.float32)

    if device is not None:
        z_A = z_A.to(device)
        z_B = z_B.to(device)

    y_A = torch.tensor(dataset_A.celltypes, dtype=torch.long, device=z_A.device)
    y_B = torch.tensor(dataset_B.celltypes, dtype=torch.long, device=z_A.device)

    n_A = z_A.shape[0]

    Z = torch.cat([z_A, z_B], dim=0)
    Y = torch.cat([y_A, y_B], dim=0)

    if normalize:
        Z = F.normalize(Z, dim=1)

    prototypes = []
    for c in range(num_classes):
        mask_c = Y == c
        if mask_c.sum() == 0:
            proto = torch.zeros(Z.size(1), device=Z.device)
        else:
            proto = Z[mask_c].mean(dim=0)
            if normalize:
                proto = F.normalize(proto, dim=0)
        prototypes.append(proto)

    prototypes = torch.stack(prototypes) 

    unlabeled_mask = Y < 0

    if unlabeled_mask.sum() == 0:
        print("No unlabeled cells found. Nothing updated.")

    Z_unlabeled = Z[unlabeled_mask]

    sim = torch.matmul(Z_unlabeled, prototypes.T)

    conf, pred = sim.max(dim=1)

    confident = conf > threshold

    idx_unlabeled = torch.where(unlabeled_mask)[0]
    idx_update = idx_unlabeled[confident]

    if len(idx_update) == 0:
        print("No confident pseudo labels assigned.")

    pseudo_labels = Y.clone()
    pseudo_labels[idx_update] = pred[confident]

    pseudo_A = pseudo_labels[:n_A].cpu().numpy()
    pseudo_B = pseudo_labels[n_A:].cpu().numpy()

    mask_A = (dataset_A.celltypes == -1) & (pseudo_A != -1)
    mask_B = (dataset_B.celltypes == -1) & (pseudo_B != -1)

    dataset_A.update_pseudo_labels(pseudo_A, mask_A)
    dataset_B.update_pseudo_labels(pseudo_B, mask_B)

    unknown_A = (y_A == -1)
    unknown_B = (y_B == -1)

    num_unknown_A = unknown_A.sum().item()
    num_unknown_B = unknown_B.sum().item()

    num_update_A = mask_A.sum().item()
    num_update_B = mask_B.sum().item()

    print(f"Updated A: {num_update_A}/{num_unknown_A} "
        f"({num_update_A / (num_unknown_A + 1e-8) * 100:.2f}%)")

    print(f"Updated B: {num_update_B}/{num_unknown_B} "
        f"({num_update_B / (num_unknown_B + 1e-8) * 100:.2f}%)")

