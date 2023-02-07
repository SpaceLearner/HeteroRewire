import os.path as osp
import sys
sys.path.append("../../src")
sys.path.append("../")

import torch

from torch_geometric.datasets import AMiner, HGBDataset
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.transforms import AddMetaPaths
from datasets import HGBDataset

import argparse

from tqdm import tqdm

def load_dataset_pyg(dataname="HGB_ACM", seed=0, device=None):
    
    if dataname[:3] == "HGB":
        
        dataset = HGBDataset(root="../../datasets/hgb", name=dataname[4:])
        
        if dataset.name == "acm":
                
            graph = dataset.data
            
            # print(graph["paper"].y[graph["paper"].test_mask][:100])

            graph["paper", "self", "paper"].edge_index = torch.arange(len(graph["paper"].x))[None, :].repeat(2, 1)
            
            metapaths = [
                [("paper", "cite", "paper"),   ("paper", "self", "paper")],
                [("paper",  "ref", "paper"),   ("paper", "self", "paper")],
                [("paper", "cite", "paper"),   ("paper",   "ref", "paper")],
                # [("paper",   "cite", "paper"), ("paper", "to", "author"),  ("author",  "to", "paper"), ("paper", "to", "subject"), ("subject", "to",   "paper"), ("paper", "to", "subject"), ("subject", "to",   "paper"), ("paper", "to", "term"), ("term",    "to",   "paper")],
                [("paper", "to", "author"), ("author",  "to",   "paper")],
                [("paper", "to", "subject"), ("subject", "to",   "paper")],
                [("paper", "to", "term"), ("term",    "to",   "paper")],
                [("paper", "cite", "paper"),   ("paper", "to", "subject"), ("subject", "to", "paper")],
                [("paper", "ref", "paper"),    ("paper", "to", "author"),  ("author",  "to", "paper")],
                # [("paper", "ref", "paper"),    ("paper", "to", "subject"), ("subject", "to", "paper")],
                # [("paper", "cite", "paper"),   ("paper", "to", "term"),    ("term",    "to", "paper")],
                # [("paper", "to",   "author"),  ("author",  "to", "paper")],
                # [("paper", "to",   "subject"), ("subject", "to", "paper")],
                # [("paper", "to",   "term"),    ("term",    "to", "paper")],
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            # print(graph)
            # AddMetaPaths()
            
            target = "paper"

        elif dataset.name == "dblp":
            
            graph = dataset.data
            
            metapaths = [
                [("author", "to", "paper"), ("paper", "to", "author")],
                [("author", "to", "paper"), ("paper",   "to", "term"), ("term", "to", "paper"), ("paper", "to", "author")],
                [("author", "to", "paper"), ("paper", "to", "venue"), ("venue", "to", "paper"), ("paper", "to", "author")],
                # [("paper",  "to", "term"),  ("term",  "to", "paper")],
                # [("term",   "to", "paper"), ("paper", "to", "term")],
                # [("venue",  "to", "paper"), ("paper", "to", "venue")]
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)

            target = "author"
            
        elif dataset.name == "imdb":
            
            graph = dataset.data

            metapaths = [
                [("movie", "to", "director"), ("director", "to", "movie")],
                # [("director", "to", "movie"), ("movie",    "to", "director")]
                [("movie", ">actorh", "actor"), ("actor", "to", "movie")],
                [("movie", "to", "keyword"), ("keyword", "to", "movie")]
               #  [("movie", "to", "director"), ("director", "to", "movie"), ("movie", ">actorh", "actor"), ("actor", "to", "movie")],
                # [("movie", "to", "director"), ("director", "to", "movie"), ("movie", "to", "keyword"),  ("keyword", "to", "movie")],
                # [("movie", ">actorh", "actor"), ("actor", "to", "movie"), ("movie", "to", "keyword"), ("keyword", "to", "movie")]
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            
            target = "movie"
            
        elif dataset.name == "freebase":
            
            graph = dataset.data
            graph["book", "self", "book"].edge_index = torch.arange(graph["book"].num_nodes)[None, :].repeat(2, 1)
            edge_types = graph.edge_types
            
            for etype in edge_types:
                graph[etype[2], "rev_"+etype[1], etype[0]].edge_index = graph[etype].edge_index[torch.LongTensor([1, 0])]
            
            metapaths = [
                [("book", "and", "book"), ("book", "self", "book")],
                [("book", "rev_about", "business"),  ("business", "about", "book")],
                [("book", "to", "film"),  ("film", "rev_to", "book")],
                [("book", "on", "location"),  ("location", "rev_on", "music"), ("music", "in", "book")],
                [("book", "about", "organization"),  ("organization", "in", "film"), ("film", "rev_to", "book")],
                [("book", "rev_to", "people"), ("people", "to", "book")],
                [("book", "rev_to", "people"), ("people", "to", "sports"), ("sports", "rev_on", "book")],
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)

            target = "book"
    
    graph = graph.to(device)

    return graph, metapaths, target

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataname = args.dataset

    data, metapaths, target = load_dataset(dataname, device=device)
    
    print(data)
    
    if args.dataset == "HGB_ACM":
        targets = ["paper", "author", "subject", "term"] 
    elif args.dataset == "HGB_DBLP":
        targets = ["author", "paper", "term", "venue"]
    elif args.dataset == "HGB_IMDB":
        targets = ["movie", "director", "actor", "keyword"]
    else:
        targets = ["book", "film", "location", "music", "person", "sport", "organization", "business"]
        
    # for idx, target in enumerate(targets):

    model = MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
                            metapath=metapaths[0], walk_length=50, context_size=7,
                            walks_per_node=5, num_negative_samples=5, sparse=True, num_nodes_dict=data.num_nodes_dict).to(device)
        
    loader = model.loader(batch_size=128, shuffle=True, num_workers=8)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train(epoch, log_steps=100, eval_steps=2000):
        model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

                # total_loss += loss.item()
                # if (i + 1) % log_steps == 0:
                #     print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                #         f'Loss: {total_loss / log_steps:.4f}'))
                #     total_loss = 0

                # if (i + 1) % eval_steps == 0:
                #     acc = test()
                #     print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                #         f'Acc: {acc:.4f}'))

    @torch.no_grad()
    def test(train_ratio=0.2):
        model.eval()

        z = model(target)
        print(model(target).shape)
        y = data[target].y

        perm = torch.randperm(z.size(0))
        train_perm = perm[:int(z.size(0) * train_ratio)]
        test_perm = perm[int(z.size(0) * train_ratio):]

        return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                        max_iter=150)

    print("training target {}".format(target))
    for epoch in tqdm(range(1, 200)):
        train(epoch)
        # åprint(test())
        
    print(model)
    zs = []
    with torch.no_grad():
        model.eval()
        for idx, target1 in enumerate(targets):
            z = model(target1)
            zs.append(z)
        zs = torch.vstack(zs)
        torch.save(zs, "checkpoints/"+dataname+".emb")
            
            
        # model.
    # acc = test()
    # print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="HGB_DBLP", help="dataset to use. ")
    # parser.add_argument("--metapath", default="HGB_ACM", help="dataset to use. ")
    # parser.add_argument("--target",   default="au", help="dataset to use. ")
    
    args = parser.parse_args()
    
    main(args)
