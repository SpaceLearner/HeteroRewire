import json
import os
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (HeteroData, InMemoryDataset, download_url,
                                  extract_zip)

from torch_geometric.utils import index_to_mask

from scipy.io import loadmat

import pandas as pd

import numpy as np

class Fraud(InMemoryDataset):

    names = {
        "amazon": 0,
        "yelp": 1,
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        print(self.names)
        assert self.name in set(self.names.keys())
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "Fraud", self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "Fraud", self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        x = [self.name + ".mat"]
        return x

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        
        pass

    def process(self):
        
        data = HeteroData()
        
        network = loadmat(self.raw_paths[0])
        
        print(network.keys())
        
        print(network["features"].shape)
        print(network["label"].shape)
        print(network["homo"].shape)
        print(network["net_upu"].shape)
        
        data["user"].num_nodes = network["features"].shape[0]
        indices = np.stack(np.nonzero(network["features"]))
        values  = network["features"].data
        data["user"].x = torch.sparse_coo_tensor(indices, values, size=network["features"].shape).float()
        data["user"].y = torch.tensor(network["label"][0], dtype=torch.long)
        randindex = torch.randperm(len(data["user"].x))
        train_index = randindex[:int(len(randindex) * 0.8)]
        test_index  = randindex[int(len(randindex) * 0.8):]
        data["user"].train_mask = index_to_mask(train_index, randindex.shape[0])
        data["user"].test_mask  = index_to_mask(test_index,  randindex.shape[0])
        
        indices = np.stack(np.nonzero(network["homo"]))
        data[("user", "to", "user")].edge_index = torch.tensor(indices, dtype=torch.long)
        indices = np.stack(np.nonzero(network["net_upu"]))
        data[("user", "p", "user")].edge_index = torch.tensor(indices, dtype=torch.long)
        indices = np.stack(np.nonzero(network["net_usu"]))
        data[("user", "s", "user")].edge_index = torch.tensor(indices, dtype=torch.long)
        indices = np.stack(np.nonzero(network["net_uvu"]))
        data[("user", "v", "user")].edge_index = torch.tensor(indices, dtype=torch.long)
        
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def __repr__(self) -> str:
        return f'{self.name}()'
    
    
if __name__ == "__main__":
    dataset = Fraud(root="../../datasets", name="amazon")
    print(dataset.data)
    
