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

from torch_geometric.utils import index_to_mask, is_undirected, subgraph
from torch_geometric.transforms import AddMetaPaths
from sklearn.feature_extraction.text import CountVectorizer


class FakeNewsNet(InMemoryDataset):
    
    def __init__(self, root: str, name: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'FakeNewsNet', 'raw', self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'FakeNewsNet', 'processed', self.name)
    
    @property
    def raw_file_names(self) -> List[str]:
        x = [self.name+"_real.csv", self.name+"_fake.csv"]
        return x

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
        
    def process(self):
        
        real_file = pd.read_csv(self.raw_paths[0])
        fake_file = pd.read_csv(self.raw_paths[1])
        
        real_file["label"] = np.zeros(len(real_file),  dtype=np.int64)
        fake_file["label"] = np.ones(len(fake_file), dtype=np.int64)
        
        file = pd.concat([real_file, fake_file], axis=0)
        
        file["tweet_ids"] = file["tweet_ids"].astype(str)
        
        unique_twiters = []
        counters       = defaultdict(lambda: 0)
        for i, twitters in enumerate(file["tweet_ids"]):
            ids = twitters.strip().split()
            for id in ids:
                unique_twiters.append(id) 
        
        counters = Counter(unique_twiters)
        unique_twiters = {}
        for key in counters:
            if counters[key] > 1:
                unique_twiters[key] = len(unique_twiters)
        
        news2twitter = []
        for idx, row in file.iterrows():
            twitters = row["tweet_ids"].strip().split()
            for twitter in twitters:
                if twitter in unique_twiters:
                    news2twitter.append([idx, unique_twiters[twitter]])

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(file["title"].values)
        indices = torch.tensor(np.stack(np.nonzero(X)), dtype=torch.long)
        values  = torch.ones(indices.shape[1])
        
        
        data = HeteroData()
        data["news"].num_nodes = X.shape[0]
        data["news"].x = torch.sparse_coo_tensor(indices, values, torch.Size([X.shape[0], X.shape[1]]))
        data["news"].y = torch.tensor(file["label"].values)
        
        data["tweet"].num_nodes = len(unique_twiters)
        indices = torch.eye(len(unique_twiters), dtype=torch.long).nonzero().T
        values  = torch.ones(len(unique_twiters))
        data["tweet"].num_nodes = len(unique_twiters)
        data["tweet"].x = torch.sparse_coo_tensor(indices=indices, values=values, size=torch.Size([len(unique_twiters), len(unique_twiters)]))
        
        data["news", "to", "tweet"].edge_index = torch.tensor(news2twitter, dtype=torch.long).T
        data["tweet", "to", "news"].edge_index = torch.flip(torch.tensor(news2twitter, dtype=torch.long).T, dims=[0])
        
        index1 = torch.randperm(len(real_file))
        index2 = torch.randperm(len(fake_file)) + len(index1)
        
        train_index1 = index1[torch.arange(len(index1))[len(index1)//5:]]
        train_index2 = index2[torch.arange(len(index2))[len(index2)//5:]]
        
        train_index  = torch.hstack([train_index1, train_index2])
        train_mask   = index_to_mask(train_index, len(file))
        test_mask    = ~train_mask
        
        data["news"].train_mask, data["news"].test_mask = train_mask, test_mask
        
        data = AddMetaPaths([[("news", "to", "tweet"), ("tweet", "to", "news")]])(data)
    
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def __len__(self) -> int:
        return self.data.x.shape
            
    def __repr__(self) -> str:
        return "FakeNewsNet()"
        
        
if __name__ == "__main__":
    
    dataset = FakeNewsNet("../../datasets", "politifact")
    
    print(dataset.data)