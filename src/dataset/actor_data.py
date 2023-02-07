import pandas as pd
import numpy as np
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional

import torch
import torch_sparse
from torch_sparse import coalesce


from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from torch_geometric.utils import index_to_mask, is_undirected

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class Actor(InMemoryDataset):
    
    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'Actor', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'Actor', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ['newmovies.txt']
        return x

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
        
    def process(self):
        data = HeteroData()
        
        node_types = {0: "starring", 1: "director", 2: "writer", 3: "movie"}
        type_nodes = {"starring": 0, "director": 1, "writer": 2, "movie": 3}
        
        nodes_dict = {"idx": defaultdict(list), "name": [], "times": [], "type": [], "words": []}
        edges      = []
        
        num_nodes_dict   = defaultdict(lambda: 0)
        with open(self.raw_paths[0], "r") as fp:
            while line := fp.readline():
                line = line.strip()
                if "*Vertices" in line:
                    total_num_nodes = int(line.split()[1])
                    continue
                if "*Edge" in line:
                    continue
                words = line.split("\t")
                if len(words) > 3:             ### process for nodes
                    nodes_dict["idx"][words[3]]  .append(int(words[0]))
                    nodes_dict["name"] .append(words[1])
                    nodes_dict["times"].append(int(words[2]))
                    nodes_dict["type"] .append(type_nodes[words[3]])
                    nodes_dict["words"].append(words[4])
                    num_nodes_dict[words[3]] += 1
                elif len(words) == 3:
                    edges.append([int(words[0]), int(words[1])])
        
            fp.close()
        
        print(nodes_dict["idx"].keys(), num_nodes_dict.keys())
        
        nodes_dict["idx"].pop("movie")
        num_nodes_dict.pop("movie")
        
        nodes = np.concatenate(list(nodes_dict["idx"].values()))
        index = np.concatenate([np.arange(num) for num in num_nodes_dict.values()], axis=0)
        nodes2idx = index[np.argsort(nodes, axis=0)]
        print(len(nodes2idx))
        vectorizer = CountVectorizer()
        
        corpus = list(map(lambda x: " ".join(x.split(";")), nodes_dict["words"]))
        
        ntypes = np.array(nodes_dict["type"])
        
        for idx, type in node_types.items():
            if type == "movie":
                continue
            nidx = ntypes == idx
            corpus_x = [corpus[i] for i in range(len(nidx)) if nidx[i] != 0]
            X_x      = vectorizer.fit_transform(corpus_x)
            feat = torch.tensor(X_x.todense(), dtype=torch.long)
            feat_idx = torch.nonzero(feat, as_tuple=True)
            values = torch.ones(feat_idx[0].shape[0])
            # print(values.shape, feat_idx[0].shape, feat.shape)
            indices = torch.stack(feat_idx)
            data[type].num_nodes = nidx.sum()
            
            data[type].x = torch.sparse_coo_tensor(indices, values, feat.size())

            if idx == 0:
                lda    = LatentDirichletAllocation(n_components=7, random_state=0)
                y      = lda.fit_transform(X_x)
                y      = np.argmax(y, axis=1)
                data[node_types[0]].y = torch.tensor(y, dtype=torch.long)
             
        label_idx   = np.arange(data["starring"].x.shape[0]) 
        train_index, test_index = train_test_split(label_idx, test_size=0.2, shuffle=True)
        train_mask, test_mask   = index_to_mask(torch.LongTensor(train_index), len(label_idx)), index_to_mask(torch.LongTensor(test_index), len(label_idx))
        
        data["starring"].train_mask, data["starring"].test_mask = train_mask, test_mask
        
        edges  = np.array(edges)
       
        idx = 0
        for i in range(3):
            for j in range(3):
                edge_idx = np.nonzero((ntypes[edges[:, 0]] == i).astype(int) * (ntypes[edges[:, 1]] == j).astype(int)) 
                if edge_idx[0].shape[0] == 0:
                    continue
                edge_index    = edges[edge_idx].T
                edge_index[0] = nodes2idx[edge_index[0]]
                edge_index[1] = nodes2idx[edge_index[1]]
                edge_index    = torch.tensor(edge_index, dtype=torch.long)
                edge_index, _ = coalesce(edge_index, None, num_nodes_dict[node_types[i]], num_nodes_dict[node_types[j]])
                print(is_undirected(edge_index))
                data[(node_types[i], "rel_{}".format(idx), node_types[j])].edge_index     = edge_index
                data[(node_types[j], "rel_{}_rev".format(idx), node_types[i])].edge_index = torch.flip(edge_index, dims=[0])
                idx += 1  
        
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def __len__(self) -> int:
        return 32483
                
    def __repr__(self) -> str:
        return "Actor()"
    
if __name__ == "__main__":
    
    dataset = Actor("../../datasets")
    data    = dataset.data
    print(data[("starring", "rel_0", "starring")].edge_index[0].max())
    print((data["starring"].y[data[("starring", "rel_0", "starring")].edge_index[0]] == data["starring"].y[data[("starring", "rel_0", "starring")].edge_index[1]]).sum() / data[("starring", "rel_0", "starring")].edge_index.shape[1])