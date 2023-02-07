import sys
sys.path.append("../src")
import argparse
import torch
import torch.optim as optim
from homophily import our_measure, edge_homophily_edge_idx

from torch_geometric.data import Data
from torch_geometric.datasets import DBLP, IMDB
from torch_geometric.utils import remove_self_loops, remove_isolated_nodes, homophily, add_self_loops, to_undirected
from torch_geometric.transforms import AddMetaPaths

from pytorch_lightning import seed_everything

from datasets import *

from models import *

from GraphLearner import ModelHandler

import argparse

import wandb

def load_dataset(dataname="HGB_ACM", seed=0, device=None):
    
    seed_everything(seed)
    
    if dataname[:3] == "HGB":
        
        dataset = HGBDataset(root="../datasets/hgb", name=dataname[4:])
        
        if dataset.name == "acm":
                
            graph = dataset.data
            
            # print(graph["paper"].y[graph["paper"].test_mask][:100])

            graph["paper", "self", "paper"].edge_index = torch.arange(len(graph["paper"].x))[None, :].repeat(2, 1)
            
            metapaths = [
                [("paper", "cite", "paper"),   ("paper", "self", "paper")],
                [("paper",  "ref", "paper"),   ("paper", "self", "paper")],
                [("paper", "cite", "paper"),   ("paper",   "ref", "paper")],
                [("paper", "cite", "paper"),   ("paper", "to", "author"),  ("author",  "to", "paper")],
                [("paper", "cite", "paper"),   ("paper", "to", "subject"), ("subject", "to", "paper")],
                [("paper", "ref", "paper"),   ("paper", "to", "author"),  ("author",  "to", "paper")],
                [("paper", "ref", "paper"),   ("paper", "to", "subject"), ("subject", "to", "paper")],
                [("paper", "cite", "paper"),   ("paper", "to", "term"),    ("term",    "to", "paper")],
                [("paper", "to",   "author"),  ("author",  "to", "paper")],
                [("paper", "to",   "subject"), ("subject", "to", "paper")],
                [("paper", "to",   "term"),    ("term",    "to", "paper")],
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            # print(graph)
            target = "paper"

        elif dataset.name == "dblp":
            
            graph = dataset.data
            
            metapaths = [
                [("author", "to", "paper"), ("paper", "to", "author")],
                [("author", "to", "paper"), ("paper",   "to", "term"), ("term", "to", "paper"), ("paper", "to", "author")],
                [("author", "to", "paper"), ("paper",   "to", "venue"), ("venue", "to", "paper"), ("paper", "to", "author")]
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            #
            # print(graph)

            target = "author"
            
        elif dataset.name == "imdb":
            
            graph = dataset.data

            metapaths = [
                [("movie", "to", "director"), ("director", "to", "movie")],
                [("movie", ">actorh", "actor"), ("actor", "to", "movie")],
                [("movie", "to", "keyword"), ("keyword", "to", "movie")],
                [("movie", "to", "director"), ("director", "to", "movie"), ("movie", ">actorh", "actor"), ("actor", "to", "movie")],
                [("movie", "to", "director"), ("director", "to", "movie"), ("movie", "to", "keyword"),  ("keyword", "to", "movie")],
                [("movie", ">actorh", "actor"), ("actor", "to", "movie"), ("movie", "to", "keyword"), ("keyword", "to", "movie")]
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            
            # print(graph)
            
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
            
            # print(graph)

            target = "book"
            
        # graph = graph.to(device)

        if dataset.name != "freebase":
            features = graph[target].x
        else:
            dim = graph[target].num_nodes
            print(dim)
            indices  = np.vstack((np.arange(dim), np.arange(dim)))
            indices  = torch.LongTensor(indices)
            values   = torch.FloatTensor(np.ones(dim))
            # print(features_list[i].shape)
            features = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
            
        labels       = graph[target].y  
        
        train_index = graph[target].train_mask.nonzero()
        val_index   = train_index[torch.randperm(len(train_index))[:len(train_index) // 5]]
        val_mask    = torch.zeros(graph[target].num_nodes)
        val_mask[val_index] = 1
        val_mask    = val_mask.bool()

        train_mask  = graph[target].train_mask
        train_mask[val_mask] = 0

        train_index = train_mask.to(device)
        val_index   = val_mask.to(device)
        test_index  = graph[target].test_mask
        test_mask   = test_index
            
        num_labels = labels.max().item()+1 if len(labels.size()) == 1 else labels.shape[1]

    return graph, target, train_mask, val_mask, test_mask, labels, num_labels


def run_exp_han(dataname="HGB_IMDB", seed=0, config=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    seed_everything(seed)

    if dataname == "HGB_ACM":
        selected_metapath = [0, 1, 3, 4, 5, 6, 8, 9]
        selected_rewire   = [4, 6, 9] 
    elif dataname == "HGB_DBLP":
        selected_metapath = [0, 1, 2]
        selected_rewire   = [1, 2]
    elif dataname == "HGB_IMDB":
        selected_metapath = [0, 1, 2, 3, 4, 5]
        selected_rewire   = [1, 2, 3, 4, 5]
    elif dataname == "HGB_Freebase":
        selected_metapath = [0, 1, 2, 3, 4, 5, 6]
        selected_rewire   = [1, 2, 3, 4, 5, 6]

    graph, target, train_mask, val_mask, test_mask, labels, num_labels = load_dataset(dataname, seed=seed)

    if dataname != "HGB_Freebase":
        x_dict = {target: graph.x_dict[target].to(device)}
    else:
        dim = graph[target].num_nodes
        indices  = np.vstack((np.arange(dim), np.arange(dim)))
        indices  = torch.LongTensor(indices)
        values   = torch.FloatTensor(np.ones(dim))
        # print(features_list[i].shape)
        features = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
        x_dict = {target: features}
        graph[target].x = features

    edge_index_dict = {}
            
    for idx, key in enumerate(graph.edge_index_dict):
        if "metapath" in key[1] and int(key[1][9:]) in selected_metapath:
            edge_index_dict[key] = graph.edge_index_dict[key]
            edge_index_dict[key] = remove_self_loops(edge_index_dict[key])[0]
            edge_index_dict[key] = to_undirected(edge_index_dict[key]).to(device)
            # edge_index_dict[key] = add_self_loops(edge_index_dict[key])[0].to(device)

    for idx, key in enumerate(edge_index_dict):
        if int(key[1][9:]) in selected_rewire:
            data                 = Data(x=graph[target].x, edge_index=edge_index_dict[key], y=graph[target].y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(device)
            if config.rewire:
                Rewirer          = ModelHandler(graph[target].x.shape[1], graph[target].y.max()+1, device=device, thres_min_deg=config.thres_min_deg, thres_min_deg_ratio=config.thres_min_deg_ratio, window_size=[config.window_size, config.window_size])
                data_new         = Rewirer(data, cat_self=False, prunning=True, thres_prunning=config.thres_prunning)
            else:
                data_new         = data
            edge_index_dict[key] = data.edge_index
            
    metadata  = ([target], [(target, "metapath_"+str(i), target) for i in selected_metapath])
    model     = HAN(x_dict[target].shape[1], config.hidden, num_labels, 8, num_layers=2, one_hot=True if dataname=="HGB_Freebase" else False, meta_data=metadata).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    test_acc, test_f1_macro, test_f1_micro = model.fit(x_dict, edge_index_dict, labels.to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device), optimizer, target)
    
    return test_acc, test_f1_macro, test_f1_micro


def main(config):
    
    dataname = config.dataset
    
    seeds = [0, 1, 2, 3, 4]
    
    results = []
    
    for idx, seed in enumerate(seeds):
        results.append(run_exp_han(dataname, seed, config))

    acc      = []
    f1_macro = []
    f1_micro = []

    for result in results:
        acc     .append(result[0])
        f1_macro.append(result[1])
        f1_micro.append(result[2])
        
    wandb.log({"acc":      np.mean(acc),      "acc_std":      np.std(acc)})
    wandb.log({"f1_macro": np.mean(f1_macro), "f1_macro_std": np.std(f1_macro)})
    wandb.log({"f1_micro": np.mean(f1_micro), "f1_micro_std": np.std(f1_micro)})
        
    print("results after rewire: ", np.mean(acc), np.std(acc), np.mean(f1_macro), np.std(f1_macro), np.mean(f1_micro), np.std(f1_micro))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset",             type=str,   default="HGB_ACM",  help="dataset to use. ")
    parser.add_argument("--thres_min_deg",       type=float, default=3.,         help="threshhold for minimum degrees. ")
    parser.add_argument("--thres_min_deg_ratio", type=float, default=-1.,        help="threshhold ratio for minimum degrees. ")
    parser.add_argument("--window_size",         type=int,   default=10000,      help="window size used to rewire. ")
    parser.add_argument("--thres_prunning",      type=float, default=0.6,        help="threshhold for edge pruning. ")
    parser.add_argument("--order_neighbors",     type=float, default=2,          help="orde of neighbors to use. ")
    parser.add_argument("--hidden",              type=float, default=256,        help="hidden size. ")
    parser.add_argument("--rewire",              type=int,   default=0,          help="whether use rewire. ")
    
    args = parser.parse_args()
    
    wandb.init(project="Hetero", entity="gjyspliter", config=args)
    
    config = wandb.config
    
    main(config)
    
    
    