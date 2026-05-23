import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from anndata import AnnData
from scipy.sparse import issparse, spmatrix
from typing import Optional, Union, Dict, Iterator
import random
import pandas as pd

class AnnDataDataset(Dataset):
    def __init__(self, adata: AnnData, input_key: Optional[str] = None, source_key: Optional[str] = None):
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an AnnData object.")
        if not hasattr(adata, "obs") or adata.obs.empty:
            raise ValueError("adata must have a non-empty 'obs' attribute.")
        if not hasattr(adata, "X"):
            raise ValueError("adata must have 'X' attribute (core expression data).")

        self.input = self._get_input_tensor(adata, input_key)
        self.link_feat = self._convert_to_tensor(adata.obsm["link_feat"]) if "link_feat" in adata.obsm else None
        self.sources = self._encode_obs_column(adata, source_key) if source_key is not None else None

    def _get_input_tensor(self, adata: AnnData, input_key: Optional[str]) -> torch.Tensor:
        if input_key is None:
            data = adata.X
        elif input_key in adata.obsm:
            data = adata.obsm[input_key]
        elif hasattr(adata, input_key):
            data = getattr(adata, input_key)
        else:
            data = adata.X
        return self._convert_to_tensor(data)

    def _encode_obs_column(self, adata: AnnData, key: str):
        if key not in adata.obs:
            raise KeyError(f"{key} not found in adata.obs.")
        
        obs_series = adata.obs[key].copy()
        
        if isinstance(obs_series.dtype, pd.CategoricalDtype):
            if 'unknown' not in obs_series.cat.categories:
                obs_series = obs_series.cat.add_categories('unknown')
            obs_series = obs_series.fillna('unknown')
        else:
            obs_series = obs_series.astype(str)
            obs_series = obs_series.fillna('unknown')
        
        cat = pd.Categorical(obs_series)
        return np.asarray(cat.codes, dtype=np.int64)

    def _convert_to_tensor(self, data: Union[np.ndarray, spmatrix]) -> torch.Tensor:
        if issparse(data):
            data = data.toarray()
        return torch.from_numpy(np.asarray(data, dtype=np.float32))

    def __len__(self) -> int:
        return self.input.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "input": self.input[idx],
            "index": torch.tensor(idx, dtype=torch.long)
        }
        if self.sources is not None:
            sample["source"] = torch.tensor(self.sources[idx], dtype=torch.long)
        if self.link_feat is not None:
            sample["link_feat"] = self.link_feat[idx]
        return sample

    @property
    def feature_shapes(self) -> Dict[str, int]:
        return {"input": self.input.shape[1]}

    @property
    def source_categories(self) -> int:
        if self.sources is None:
            return 1
        valid_sources = self.sources[self.sources != -1]
        return len(np.unique(valid_sources)) if len(valid_sources) > 0 else 1

class InfiniteRandomSampler(Sampler):
    def __init__(self, data_source: Dataset, batch_size: int) -> None:
        self.sample_num = len(data_source)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            yield random.choices(range(self.sample_num), k=self.batch_size)

    def __len__(self) -> int:
        return 10**10

def load_data(dataset: Dataset, batch_size: int) -> DataLoader:
    sampler = InfiniteRandomSampler(dataset, batch_size)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)