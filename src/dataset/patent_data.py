import pandas as pd
import numpy as np
import os.path as osp
from collections import defaultdict, Counter
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from sklearn.model_selection import train_test_split
from torch_geometric.utils import index_to_mask, remove_self_loops

class Patent(InMemoryDataset):

    def __init__(self, root: str, year: int, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):

        self.year = year
        assert self.year >= 1963 and self.year <= 1999
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'Patent', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'Patent', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ["apat63_99.txt", "cite75_99.txt", "ainventor.txt"]
        return x

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):

        data = HeteroData()

        df_attr    = pd.read_csv(self.raw_paths[0])
        edges      = pd.read_csv(self.raw_paths[1])
        inventer   = pd.read_csv(self.raw_paths[2])
        df_attr    = df_attr[df_attr["GYEAR"].isin(list(range(self.year-4, self.year+1)))]
        df_seeds   = df_attr[df_attr["GYEAR"] == self.year]
        patents_unique = set(df_attr["PATENT"].values.tolist())

        print(len(patents_unique))

        edges_ori      = edges[edges["CITING"].isin(patents_unique)]
        edges_ori      = edges[edges["CITED"] .isin(patents_unique)]

        df_index   = np.random.permutation(len(df_seeds))
        patents    = df_seeds["PATENT"].values[df_index[:2000]]
        nodes_from = set(patents.tolist())
        edges      = edges_ori[edges_ori["CITING"].isin(nodes_from)]

        nodes_to   = edges["CITED"].values

        nodes      = np.concatenate([np.array(list(nodes_from)), nodes_to], axis=0)
        nodes      = np.unique(nodes)

        nodes_set  = set(nodes.tolist())

        edges      = edges_ori[edges_ori["CITING"].isin(nodes_set)]
        edges      = edges[edges["CITED"] .isin(nodes_set)]
        
        node2id         = {node:i for i, node in enumerate(nodes)}
        edges["CITING"] = edges["CITING"].map(node2id)
        edges["CITED"]  = edges["CITED"] .map(node2id)

        print(edges.head())

        nodes_attr = df_attr[df_attr["PATENT"].isin(nodes)]

        nodes_attr["PATENT"] = nodes_attr["PATENT"].map(node2id)
        nodes_attr = nodes_attr.sort_values(by='PATENT')

        nodes_attr = nodes_attr[["PATENT", "GYEAR", "COUNTRY", "CLAIMS", "NCLASS", "CAT", "SUBCAT", "CRECEIVE"]]
        claims_unique = nodes_attr["CLAIMS"].unique()
        claims2id     = {claim:i for i, claim in enumerate(claims_unique)}
        nodes_attr["CLAIMS"] = nodes_attr["CLAIMS"].map(claims2id)
        cat_unique    = nodes_attr["CAT"].unique()
        cat2id        = {cat:i for i, cat in enumerate(cat_unique)}
        nodes_attr["CAT"] = nodes_attr["CAT"].map(cat2id)
        subcat_unique = nodes_attr["SUBCAT"].unique()
        subcat2id     = {subcat:i for i, subcat in enumerate(subcat_unique)}
        nodes_attr["SUBCAT"] = nodes_attr["SUBCAT"].map(subcat2id)

        features1     = (np.arange(len(claims_unique)) == nodes_attr["CLAIMS"].values[:, None].astype(np.int64)).astype(np.int64)
        features2     = (np.arange(len(cat_unique))    == nodes_attr["CAT"].values[:, None].astype(np.int64))   .astype(np.int64)
        features3     = (np.arange(len(subcat_unique)) == nodes_attr["SUBCAT"].values[:, None].astype(np.int64)).astype(np.int64)

        features      = np.concatenate([features1, features2, features3], axis=1)

        feat_index  = np.stack(np.nonzero(features))
      
        values = torch.ones(feat_index.shape[1])
        data["patent"].x = torch.sparse_coo_tensor(indices=torch.tensor(feat_index, dtype=torch.long), values=values, size=features.shape)
        data[("patent", "citing", "patent")].edge_index = remove_self_loops(torch.tensor(edges.values.T, dtype=torch.long))[0]
        data[("patent", "cited",  "patent")].edge_index = torch.flip(data[("patent", "citing", "patent")].edge_index, dims=[0])

        inventer = inventer[inventer["PATENT"].isin(nodes)]

        inventer["PATENT"]   = inventer["PATENT"].map(node2id)
        inventer["LASTNAM"]  = inventer["LASTNAM"].astype(str)
        inventer["FIRSTNAM"] = inventer["FIRSTNAM"].astype(str)
        inventer["MIDNAM"]   = inventer["MIDNAM"].astype(str)
        inventer["CITY"]     = inventer["CITY"].astype(str)
        
        values = list(map(lambda x: " ".join(x), inventer[["LASTNAM", "FIRSTNAM", "MIDNAM", "CITY"]].values))
        inventer["INVENTER"] = values
        inventer_unique = inventer["INVENTER"].unique()
        inventer2id = {inventer:i for i, inventer in enumerate(inventer_unique)}
        inventer["INVENTER"] = inventer["INVENTER"].map(inventer2id)

        print(inventer["INVENTER"].shape)

        data["inventer"].num_nodes = len(inventer_unique)
        indices = torch.eye(len(inventer_unique)).nonzero().T.float()
        print(indices.shape)
        values  = torch.ones(len(inventer_unique))
        data["inventer"].x = torch.sparse_coo_tensor(indices=indices, values=values, size=torch.eye(len(inventer_unique)).shape)

        data[("patent", "to", "inventer")].edge_index = torch.tensor(inventer[["PATENT", "INVENTER"]].values.T, dtype=torch.long)
        data[("inventer", "to", "patent")].edge_index = torch.flip(data[("patent", "to", "inventer")].edge_index, dims=[0])

        print(nodes_attr.head())
        country_unique = nodes_attr["COUNTRY"].unique()
        country2id     = {country:i for i, country in enumerate(country_unique)}
        nodes_attr["COUNTRY"] = nodes_attr["COUNTRY"].map(country2id)
        # nodes_attr["COUNTRY"] = nodes_attr["COUNTRY"].astype(np.int64)

        # print(nodes_attr["COUNTRY"])

        data["country"].num_nodes = len(country_unique)
        indices = torch.eye(len(country_unique)).nonzero().T.float()
        values  = torch.ones(len(country_unique))
        data["country"].x = torch.sparse_coo_tensor(indices=indices, values=values, size=torch.eye(len(country_unique)).shape)
        
        data[("patent", "to", "country")].edge_index = torch.tensor(nodes_attr[["PATENT", "COUNTRY"]].values.T, dtype=torch.long)
        data[("country", "to", "patent")].edge_index = torch.flip(data[("patent", "to", "country")].edge_index, dims=[0])

        y_unique = nodes_attr["GYEAR"].unique()
        y2id     = {y:i for i, y in enumerate(y_unique)}
        nodes_attr["GYEAR"] = nodes_attr["GYEAR"].map(y2id)

        data["patent"].y = torch.tensor(nodes_attr["GYEAR"].values, dtype=torch.long)

        label_idx   = np.arange(data["patent"].x.shape[0]) 
        train_index, test_index = train_test_split(label_idx, test_size=0.2, shuffle=True)
        train_mask, test_mask   = index_to_mask(torch.LongTensor(train_index), len(label_idx)), index_to_mask(torch.LongTensor(test_index), len(label_idx))
        
        data["patent"].train_mask, data["patent"].test_mask = train_mask, test_mask

        torch.save(self.collate([data]), self.processed_paths[0])

    def __len__(self) -> int:
        return self.data["patent"].num_nodes + self.data["country"].num_nodes + self.data["inventer"].num_nodes
  
    def __repr__(self) -> str:
        return f'{self.year}()'

if __name__ == "__main__":
    dataset = Patent(root="../../datasets/", year=1998)
    print(dataset.data)
