import pandas as pd
import numpy as np
import os.path as osp
from collections import defaultdict, Counter
from typing import Callable, List, Optional

from datasets import load_dataset

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
import scipy

class Liar(InMemoryDataset):
    
    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'Liar', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'Liar', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = [None]
        return x

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
        
    def process(self):
        
        
        dataset = load_dataset("liar")
        
        
        data = defaultdict(list)

        for phrase in ["train", "validation", "test"]:
            for idx, sample in enumerate(dataset[phrase]):
                data["idx"]    .append(sample["id"])
                data["label"]  .append(sample["label"])
                data["feat"]   .append(sample["statement"])
                data["subject"].append(sample["subject"])
                data["speaker"].append(sample["speaker"])
                data["context"].append(sample["context"])
                data["phrase"] .append(phrase)
                
        df = pd.DataFrame(data)
        
        num_nodes = len(set(df["idx"].values))

        thresh = 2

        unique_speakers = Counter(df["speaker"].values)

        speakers2id     = {}
        speakers_remain = {}

        for key in unique_speakers:
            speakers2id[key] = len(speakers2id)
            if unique_speakers[key] > thresh:
                speakers_remain[speakers2id[key]] = len(speakers_remain)

        # df["speaker"] = df["speaker"].map(speakers2id)

        unique_subjects = {}
        for subjects in df["subject"]:
            words = subjects.strip().split(",")
            for word in words:
                if word not in unique_subjects:
                    unique_subjects[word] = 1
                else:
                    unique_subjects[word] += 1
            
        subjects2id     = {}
        subjects_remain = {}

        for key in unique_subjects:
            subjects2id[key] = len(subjects2id)
            if unique_subjects[key] > thresh:
                subjects_remain[subjects2id[key]] = len(subjects_remain)
                
        unique_context = Counter(df["context"].values)

        context2id     = {}
        context_remain = {}

        for key in unique_context:
            context2id[key] = len(context2id)
            if unique_context[key] > thresh:
                context_remain[context2id[key]] = len(context_remain)
                
        feats = df["feat"].values.tolist()

        vectorizer = CountVectorizer()
        
        # X = vectorizer.fit_transform(feats)
        
        X = scipy.sparse.eye(len(feats))

        data = HeteroData()

        indices = torch.stack([torch.tensor(X.tocoo().row), torch.tensor(X.tocoo().col)]).long()
        values = torch.ones(X.tocoo().row.shape[0])
        data["news"].num_nodes = num_nodes
        data["news"].x = torch.sparse_coo_tensor(indices=indices, values=values, size=torch.Size([X.shape[0], X.shape[1]]))
        data["news"].y = torch.tensor(df["label"], dtype=torch.long)
        
        train_mask, test_mask = torch.zeros(len(data["news"].x)).long().bool(), torch.zeros(len(data["news"].x)).long().bool()
        
        train_mask[df["phrase"] == "train"] = 1
        train_mask[df["phrase"] == "validation"] = 1
        test_mask[df["phrase"]  == "test"] = 1
        
        data["news"].train_mask, data["news"].test_mask = train_mask, test_mask

        indices = torch.eye(len(speakers_remain), dtype=torch.long).nonzero().T
        values  = torch.ones(len(speakers_remain))
        data["speaker"].num_nodes = len(speakers_remain)
        data["speaker"].x = torch.sparse_coo_tensor(indices=indices, values=values, size=torch.Size([len(speakers_remain), len(speakers_remain)]))

        indices = torch.eye(len(subjects_remain), dtype=torch.long).nonzero().T
        values  = torch.ones(len(subjects_remain))
        data["subject"].num_nodes = len(subjects_remain)
        data["subject"].x = torch.sparse_coo_tensor(indices=indices, values=values, size=torch.Size([len(subjects_remain), len(subjects_remain)]))

        indices = torch.eye(len(context_remain), dtype=torch.long).nonzero().T
        values  = torch.ones(len(context_remain))
        data["context"].num_nodes = len(context_remain)
        data["context"].x = torch.sparse_coo_tensor(indices=indices, values=values, size=torch.Size([len(context_remain), len(context_remain)]))

        news_to_speaker = []
        news_to_subject = []
        news_to_context = []

        for idx, row in df.iterrows():
            if speakers2id[row["speaker"]] in speakers_remain:
                news_to_speaker.append([idx, speakers_remain[speakers2id[row["speaker"]]]])
            words = row["subject"].strip().split(",")
            for word in words:
                if subjects2id[word] in subjects_remain:
                    news_to_subject.append([idx, subjects_remain[subjects2id[word]]])
            if context2id[row["context"]] in context_remain:
                news_to_context.append([idx, context_remain[context2id[row["context"]]]])
                
        data[("news",    "to",  "speaker")].edge_index = torch.tensor(news_to_speaker, dtype=torch.long).T
        data[("speaker", "to", "news")].edge_index = torch.flip(torch.tensor(news_to_speaker, dtype=torch.long).T, dims=[0])
        data[("news",    "to",  "subject")].edge_index = torch.tensor(news_to_subject, dtype=torch.long).T
        data[("subject", "to", "news")].edge_index = torch.flip(torch.tensor(news_to_subject, dtype=torch.long).T, dims=[0])
        data[("news",    "to",  "context")].edge_index = torch.tensor(news_to_context, dtype=torch.long).T
        data[("context", "to", "news")].edge_index = torch.flip(torch.tensor(news_to_context, dtype=torch.long).T, dims=[0])
        
        print(data[("news",    "to",  "speaker")].edge_index)
        print(data[("speaker", "to", "news")].edge_index)

        #  data = AddMetaPaths([[("news", "to", "context"), ("context", "to", "news")]])(data)
        
        # subset_dict = {"news": torch.arange(X.shape[0]), \
        #                "speaker": torch.arange(data["speaker"].num_nodes), \
        #                 "subject": torch.arange(data["subject"].num_nodes)}

        torch.save(self.collate([data]), self.processed_paths[0])
        
        def __len__(self) -> int:
            return 12682
                
        def __repr__(self) -> str:
            return "Liar()"
    
if __name__ == "__main__":
    
    dataset = Liar("../../datasets")
    data    = dataset.data
    
    print(data)
    
    # metapaths = [[("news", "to", "speaker"), ("speaker", "to", "news")],
    #              [("news", "to", "subject"), ("subject", "to", "news")],
    #              [("news", "to", "context"), ("context", "to", "news")]]
    
    # data    = AddMetaPaths(metapaths)(data)
    # print(data[("news", "metapath_0", "news")].edge_index[0].max())
    # print((data["news"].y[data[("news", "metapath_2", "news")].edge_index[0]] == data["news"].y[data[("news", "metapath_2", "news")].edge_index[1]]).sum() / data[("news", "metapath_2", "news")].edge_index.shape[1])
    
    torch.save(data["news"].x, "../saved_embeds/Liar.embd")