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

class FBDataset(InMemoryDataset):
    r"""A variety of heterogeneous graph benchmark datasets from the
    `"Are We Really Making Much Progress? Revisiting, Benchmarking, and
    Refining Heterogeneous Graph Neural Networks"
    <http://keg.cs.tsinghua.edu.cn/jietang/publications/
    KDD21-Lv-et-al-HeterGNN.pdf>`_ paper.

    .. note::
        Test labels are randomly given to prevent data leakage issues.
        If you want to obtain final test performance, you will need to submit
        your model predictions to the
        `HGB leaderboard <https://www.biendata.xyz/hgb/>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"ACM"`,
            :obj:`"DBLP"`, :obj:`"Freebase"`, :obj:`"IMDB"`)
        transform (callable, optional): A function/transform that takes in an
            :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://cloud.tsinghua.edu.cn/d/2d965d2fc2ee41d09def/files/'
           '?p=%2F{}.zip&dl=1')

    names = {
        "American75": 0,
        "MIT8": 1,
        "Harvard1": 2,
        "Amherst41": 3
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        print(self.names)
        assert self.name in set(self.names.keys())
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "facebook100")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed", self.name)

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
        
        graph    = network["A"]
        metadata = network["local_info"]
        
        local_info_names = {0:"status", 1:"major", 2:"second major", 3:"house", 4:"year", 5:"high school"}
        y = metadata[:, 1]
        metadata = np.concatenate([metadata[:, :1], metadata[:, 2:]], axis=1)
        
        n_types = {0: "person", 1: "status", 2: "major", 3: "second major", 4: "house", 5: "year", 6: "high school"}
        items2idxes = []

        for i in range(metadata.shape[1]):
            items     = np.unique(metadata[:, i]).astype(np.int64)[1:]
            print(items[:10])
            items2idx = {items[idx]:idx for idx in range(len(items))}
            items2idxes.append(items2idx)

        df = pd.DataFrame({local_info_names[idx]: metadata[:, idx] for idx in range(len(local_info_names))})

        for i in range(len(items2idxes)):
            
            df[local_info_names[i]] = df[local_info_names[i]].map(items2idxes[i])
            
            print(df[local_info_names[i]].head())
                
        pos = [0] + [len(_) for idx, _ in enumerate(items2idxes[:-1]) if idx]
        pos = [0] + [_ + len(df) for _ in pos]
        
        print(len(items2idxes))
        
        for idx, n_type in n_types.items():
            if n_type == "person":
                data[n_type].num_nodes = graph.shape[0]
                indices = torch.arange(graph.shape[0])[None, :].repeat(2, 1)
                values  = torch.ones(graph.shape[0])
                data[n_type].x = torch.sparse_coo_tensor(indices, values)
            else:
                data[n_type].num_nodes = len(items2idxes[idx-1])
                indices = torch.arange(len(items2idxes[idx-1]))[None, :].repeat(2, 1)
                values  = torch.ones(len(items2idxes[idx-1]))
                data[n_type].x = torch.sparse_coo_tensor(indices, values)
                
        # x_dict = defaultdict(list)
        person_nodes = torch.arange(graph.shape[0])
        for idx, n_type in n_types.items():
            if n_type != "person":
                valid_index = (df[n_type] != df[n_type])
                valid_index = torch.tensor(valid_index, dtype=torch.long)
                e_type = ("person", "to", n_type)
                temp = df[n_type].values[~valid_index.numpy().astype(np.bool)]
                print(temp.shape)
                data[e_type].edge_index = torch.stack([person_nodes[~valid_index.bool()], torch.tensor(temp, dtype=torch.long)])
                e_type = (n_type, "rev_to", "person")
                data[e_type].edge_index = torch.stack([torch.tensor(temp, dtype=torch.long), person_nodes[~valid_index.bool()]])
        
        edge_index = np.stack(graph.nonzero())
        data[("person", "to", "person")].edge_index = torch.tensor(edge_index, dtype=torch.long)
        # data[("person", "rev_to", "person")].edge_index = torch.flip(torch.tensor(edge_index, dtype=torch.long), dims=[0])
        
        label_index  = np.arange(graph.shape[0])[y != 0]
        label_index  = torch.tensor(label_index, dtype=torch.long)
        
        select_index = torch.randperm(len(label_index))
        train_index  = label_index[select_index[:int(len(label_index)*0.8)]]
        test_index   = label_index[select_index[int(len(label_index)*0.8):]]
        
        data["person"].train_mask = index_to_mask(train_index, graph.shape[0])
        data["person"].test_mask  = index_to_mask(test_index,  graph.shape[0])
        
        data["person"].y = torch.tensor(y.astype(np.int64)-1, dtype=torch.long)
        
        data.pos = torch.tensor(pos, dtype=torch.long)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
    
    
if __name__ == "__main__":
    dataset = FBDataset(root="../../datasets/Facebook", name="American75")
    print(dataset.data)
    
