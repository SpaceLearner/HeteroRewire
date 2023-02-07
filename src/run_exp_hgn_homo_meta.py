import sys
from unittest import findTestCases

from utils.data import load_data
sys.path.append("../src")
import argparse
import torch
import torch.optim as optim
from homophily import our_measure, edge_homophily_edge_idx

from torch_geometric.data import Data
from torch_geometric.datasets import DBLP, IMDB
from torch_geometric.utils import remove_self_loops, remove_isolated_nodes, homophily, add_self_loops, to_undirected, index_to_mask
from torch_geometric.transforms import AddMetaPaths

from pytorch_lightning import seed_everything

from dataset import *

from models_dgl import Trainer

from GraphLearner import ModelHandler

import argparse
import copy
from torch_sparse import coalesce

import networkx as nx

import wandb

import dgl

def load_dataset_pyg(dataname="HGB_ACM", seed=0, device=None):
    
    seed_everything(seed)
    
    if dataname[:3] == "HGB":
        
        dataset = HGBDataset(root="../datasets/hgb", name=dataname[4:])
        
        if dataset.name == "acm":
                
            graph = dataset.data
            
            # print(graph["paper"].y[graph["paper"].test_mask][:100])

            graph["paper", "self", "paper"].edge_index = torch.arange(len(graph["paper"].x))[None, :].repeat(2, 1)
            
            metapaths = [
                [("paper", "cite", "paper"),  ("paper", "self", "paper")],
                [("paper", "ref", "paper"),   ("paper", "self", "paper")],
                # [("paper", "to", "author")],
                # [("paper", "cite", "paper") ,   ("paper",   "ref", "paper")],
                [("paper", "cite", "paper"),   ("paper", "to", "author"),  ("author",  "to", "paper")],
                [("paper", "cite", "paper"),   ("paper", "to", "subject"), ("subject", "to", "paper")],
                # [("paper", "ref", "paper"),   ("paper", "to", "author"),  ("author",  "to", "paper")],
                # [("paper", "ref", "paper"),   ("paper", "to", "subject"), ("subject", "to", "paper")],
                # [("paper", "cite", "paper"),   ("paper", "to", "term"),    ("term",    "to", "paper")],
                [("paper", "to",   "author"),  ("author",  "to", "paper")],
                [("paper", "to",   "subject"), ("subject", "to", "paper")],
                # [("paper", "to",   "term"),    ("term",    "to", "paper")],
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
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
            
            target = "movie"
            
            graph = dataset.data
            
            graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)

            metapaths = [[("movie", "to", "director"),   ("director", "to", "movie")], [("movie", ">actorh", "actor"), ("actor", "to", "movie")]]
                # [("movie", "to", "director"),   ("director", "to", "movie"), ("movie", ">actorh", "actor"), ("actor", "to", "movie")],
                # [("movie", "to", "director"),   ("director", "to", "movie"), ("movie", "to", "keyword"),    ("keyword", "to", "movie")],
                # [("movie", ">actorh", "actor"), ("actor", "to", "movie"),    ("movie", "to", "keyword"),    ("keyword", "to", "movie")]
            
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            
            # print(graph)
            
        elif dataset.name == "freebase":
            
            graph = dataset.data
            graph["book", "self", "book"].edge_index = torch.arange(graph["book"].num_nodes)[None, :].repeat(2, 1)
            edge_types = graph.edge_types
            
            for etype in edge_types:
                graph[etype[2], "rev_"+etype[1], etype[0]].edge_index = graph[etype].edge_index[torch.LongTensor([1, 0])]
            
            metapaths = [
                [("book", "and", "book"), ("book", "self", "book")]]
                # [("book", "rev_about", "business"),  ("business", "about", "book")],
                # [("book", "to", "film"),  ("film", "rev_to", "book")],
                # [("book", "on", "location"),  ("location", "rev_on", "music"), ("music", "in", "book")],
                # [("book", "about", "organization"),  ("organization", "in", "film"), ("film", "rev_to", "book")],
                # [("book", "rev_to", "people"), ("people", "to", "book")],
                # [("book", "rev_to", "people"), ("people", "to", "sports"), ("sports", "rev_on", "book")],
            
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            
            # print(graph)

            target = "book"
            
        # graph = graph.to(device)
        
        print(graph.node_types)
        print(graph.edge_types)

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
            
    elif dataname[:2] == "FB":
        
        target = "person"
        
        dataset = FBDataset(root="../datasets/Facebook", name=dataname[3:])
        
        graph = dataset.data
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
        metapaths = [[("person", "to", "person"), ("person", "self", "person")], [("person", "to", "house"), ("house", "rev_to", "person")]]
    
        graph = AddMetaPaths(metapaths=metapaths)(graph)
    
    elif dataname == "Actor":
        
        target = "starring"
        
        dataset = Actor(root="../datasets/")
        
        graph = dataset.data
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
        metapaths = [[("starring", "rel_0", "starring"), ("starring", "self",      "starring")],
                    [("starring", "rel_2",   "writer"), ("writer",   "rel_2_rev", "starring")]]
                    # [("starring", "rel_1", "director"), ("director", "rel_1_rev", "starring")]]
        
        #[[("starring", "rel_2",   "writer"), ("writer",   "rel_2_rev", "starring")],
                    #  [("starring", "rel_0", "starring"), ("starring", "self",      "starring")],
                     #[("starring", "rel_1", "director"), ("director", "rel_1_rev", "starring")]]
                    
        

        graph = AddMetaPaths(metapaths=metapaths)(graph)
        
    elif dataname[:4] == "Fake":
        
        target = "news"
        
        dataset = FakeNewsNet(root="../datasets/", name=dataname[5:])
        
        graph = dataset.data
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
    elif dataname == "Liar":
        
        target = "news"
        
        dataset = Liar(root="../datasets/")
        
        graph = dataset.data
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
        metapaths = [[("news", "to", "context"), ("context", "to", "news")],
                     [("news", "to", "speaker"), ("speaker", "to", "news")]]
                    #  [("news", "to", "subject"), ("subject", "to", "news")]]
        
        # [[("news", "to", "context"), ("context", "to", "news")]]
                    #  [("news", "to", "speaker"), ("speaker", "to", "news")],
                    #  [("news", "to", "subject"), ("subject", "to", "news")]]
        
        graph = AddMetaPaths(metapaths=metapaths)(graph)
    
    elif dataname[:2] == "FD":
        
        target = "user"
        
        dataset = Fraud(root="../datasets/", name=dataname[3:])
        
        graph = dataset.data
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
        if dataname[3:] == "Amazon":
            metapaths = [[("user", "p", "user"), ("user", "self", "user")]]
                         # [("user", "to", "speaker"), ("speaker", "to", "news")]]
                         
        graph = AddMetaPaths(metapaths=metapaths)(graph)
    
    print(graph[target].y.shape)
    labels      = graph[target].y  
    
    train_index = graph[target].train_mask.nonzero()
    val_index   = train_index[torch.randperm(len(train_index))[:len(train_index) // 4]]
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


def run_exp_han(dataname="HGB_ACM", seed=0, config=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    seed_everything(seed)
    
    print(dataname)

    if dataname == "HGB_ACM":
        selected_metapath = [0, 1, 3, 4, 5, 6, 8, 9]
        selected_rewire   = [2, 3, 4, 5] 
    elif dataname == "HGB_DBLP":
        selected_metapath = [0, 1, 2]
        selected_rewire   = [0, 1, 2]
    elif dataname == "HGB_IMDB":
        selected_metapath = [0, 1, 2, 3, 4, 5]
        selected_rewire   = [0, 1, 2, 3, 4, 5]
    elif dataname == "HGB_Freebase":
        selected_metapath = [0, 1, 2, 3, 4, 5, 6]
        selected_rewire   = [0, 1, 2, 3, 4, 5, 6]
    elif dataname[:2] == "FB":
        selected_metapath = [0, 1, 2]
        selected_rewire   = [0, 1, 2]

    graph, target, train_mask, val_mask, test_mask, labels, num_labels = load_dataset_pyg(dataname, seed=seed)
    
    if dataname[:3] == "HGB":
        feat_distribution  = torch.load("experiments/checkpoints/" + config.dataset     + ".embd")
        feat_distribution  = torch.hstack(feat_distribution)
        features_list      = torch.load("experiments/checkpoints/" + config.dataset[4:] + "_feat.list")
    elif dataname[:2] == "FB":
        feat_distribution  = torch.load("experiments/checkpoints/" + config.dataset     + ".embd")
        feat_distribution  = torch.hstack(feat_distribution)
        # print(feat_distribution.shape[0])
        features_list      = [graph[n_type].x.to(device) for n_type in graph.node_types]
        # feat_distribution  = features_list[0]
    else:
        features_list      = [graph[n_type].x.to(device) for n_type in graph.node_types]
        feat_distribution  = features_list[0]
        
    print(graph)
    
    num_nodes       = [feat.shape[0] for feat in features_list]
    # # print(features_list[0].shape)
    pos             = torch.cumsum(torch.tensor([0] + [feat.shape[0] for feat in features_list[:-1]], dtype=torch.long), dim=0)# torch.cumsum(torch.tensor([feat.shape[0] for feat in features_list]), dim=0)
    pos             = {graph.node_types[idx]: pos[idx].item() for idx in range(len(graph.node_types))}
    
    in_dims = [feat.shape[1] for feat in features_list]

    graph = graph.to(device)
    # print(feat_distribution.shape)
    
    print(graph)
    
    edge_index_dict = copy.deepcopy(graph.edge_index_dict)
    edge_index_dict.pop((target, "self", target))
    for key in list(edge_index_dict.keys()):
        if "metapath" in key[1]:
            edge_index_dict.pop(key)
    for key in list(edge_index_dict.keys()):
        if key[0] == target and key[0] == key[2]:
            edge_index_dict.pop(key)
    
    edge_index_dict_rewire = copy.deepcopy(graph.edge_index_dict)
    for key in list(edge_index_dict_rewire.keys()):
        if "metapath" not in key[1]:
            edge_index_dict_rewire.pop(key)
    # if dataname == "Actor":
    #     rel = "rel_0"
    # elif dataname[:2] == "FB":
    #     rel = "to"
    # else:
    #     rel = "metapath_0"
    
    
    # edge_index_dict[(target, rel, target)] = to_undirected(edge_index_dict[(target, rel, target)])
    # if dataname == "Actor":
    #     edge_index_dict.pop((target, "rel_0_rev", target))
    
    print(edge_index_dict_rewire.keys())
    
    if config.rewire_whole:
        
        if config.dataset in ["Actor", "Liar", "Fake_politifact", "Fake_gossipcop", "FD_Amazon"]:
                feat_distribution = feat_distribution.to_dense()
        thres_prunning = [-1.0, 0.95, 0.95]
        cnt = 0
        for key, edge_index_rewire in edge_index_dict_rewire.items(): 
            print("metapath: {}, homophily ratio is {}".format(key, homophily(edge_index_rewire, y=graph[target].y)))
            rewirer = ModelHandler(feat_distribution.shape[1], num_labels, device=device, window_size=[config.window_size, config.window_size], num_epoch=200, shuffle=[True, True], thres_min_deg=config.thres_min_deg, thres_min_deg_ratio=config.thres_min_deg_ratio, use_clf=config.use_clf)
        # print(train_mask.shape, val_mask.shape, test_mask.shape, feat_distribution.shape, graph.edge_index_dict[(target, "rel_0", target)].max(), graph[target].y.shape)
            data       = Data(x=feat_distribution, num_target=len(features_list[0]), edge_index=edge_index_rewire, y=graph[target].y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(device)
            data_new   = rewirer(copy.deepcopy(data), cat_self=True, prunning=True, thres_prunning=thres_prunning[cnt])
            cnt += 1
            print("metapath: {}, after rewire homophily ratio is {}".format(key, homophily(data_new.edge_index, y=graph[target].y)))
            # edges_add.append(to_undirected(data_new.edge_index).to(device))
            edge_index_dict[(target, "metapath_"+str(cnt), target)] = to_undirected(data_new.edge_index).to(device)
            
    # else:
    #     for key, edge_index_rewire in edge_index_dict_rewire.items(): 
    #         edge_index_dict[key] = edge_index_rewire.to(device)
        # for edge_index in edges_add:
        #     print(edge_index.shape)
        # edge_index_dict[(target, "rewire", target)]     = coalesce(torch.hstack(edges_add), None, m=num_nodes[0], n=num_nodes[0])[0]
        # edge_index_dict.pop((target, "rel_0_rev", target))
        # edge_index_dict[(target, "rel_0_rev", target)] = to_undirected(data_new.edge_index).to(device)[1]
    

    for key in edge_index_dict.keys():
        edge_index_dict[key][0] += pos[key[0]]
        edge_index_dict[key][1] += pos[key[2]]

    # del graph
    if config.whole_graph:
        edge_index_dict["self"] = torch.stack([torch.arange(sum(num_nodes), device=device), torch.arange(sum(num_nodes), device=device)])
    else:
        edge_index_dict["self"] = torch.stack([torch.arange(num_nodes[0], device=device), torch.arange(num_nodes[0], device=device)])
    
    print(edge_index_dict.keys())
    
    edge_index_new = list(edge_index_dict.values())
    
    # print(edge_index_dict["new"].shape)
    
    print([key for key, _ in edge_index_dict.items()])
    print([index.shape for index in edge_index_new])
            
    e_feat = []

    num_etypes = 0
    for idx, edge_index in enumerate(edge_index_new):
        num_etypes += 1
        e_feat.append(idx * torch.ones(edge_index.shape[1], dtype=torch.long))
        
    edge_index_new = torch.hstack(edge_index_new).to(device)
    
    # if config.rewire_whole:
    #     print("Homophily Ratio After Rewire:",  homophily(edge_index_dict[(target, "rewire", target)], y=graph[target].y))
    
    e_feat = torch.hstack(e_feat).to(device)
    
    print(edge_index_new.shape, e_feat.shape, e_feat[:100])
    
    print(edge_index_new.max(), sum(num_nodes))
    
    if config.whole_graph:
        edge_index_new, e_feat = coalesce(edge_index_new, e_feat, m=sum(num_nodes), n=sum(num_nodes), op="min")
    else:
        edge_index_new, e_feat = coalesce(edge_index_new, e_feat, m=num_nodes[0], n=num_nodes[0], op="min")
    
    print(edge_index_new.shape, e_feat.shape, e_feat[:100])
    
    
    del graph
    labels = labels.to(device)
    # print(edge_index_new.shape, e_feat.max()+1)
    if config.whole_graph:
        g = dgl.DGLGraph(num_nodes=sum(num_nodes)).to(device)
    else:
        g = dgl.DGLGraph(num_nodes=num_nodes[0]).to(device)
        
    g.add_edges(edge_index_new[0], edge_index_new[1], {'e_feat': e_feat})
    e_feat = g.edata['e_feat']
    print(e_feat.max()+1)
    print(in_dims, num_labels)

    if config.use_meta_feat:
        in_dims[0] = feat_distribution.shape[1]
    
    trainer = Trainer(config.model, g, in_dims, num_labels, device, config).to(device)
    
    train_index = train_mask.nonzero().squeeze()
    val_index   = val_mask.nonzero().squeeze()
    test_index  = test_mask.nonzero().squeeze()
    
    if config.whole_graph:
        if config.use_meta_feat:
            features_list[0] = feat_distribution.to(device)
        test_f1_macro, test_f1_micro = trainer.fit(features_list, e_feat, labels, train_index, val_index, test_index, True)
    else:
        if config.use_meta_feat:
            features_list[0] = feat_distribution.to(device)
        test_f1_macro, test_f1_micro = trainer.fit([features_list[0]], e_feat, labels, train_index, val_index, test_index, True)
    
    return test_f1_macro, test_f1_micro


def main(args):
    
    dataname = args.dataset
    seeds = [0, 1, 2, 3, 4]
    results = []
    for idx, seed in enumerate(seeds):
        results.append(run_exp_han(dataname, seed, config=args))

    # acc      = []
    f1_macro = []
    f1_micro = []

    for result in results:
        f1_macro.append(result[0])
        f1_micro.append(result[1])

    wandb.log({"f1_macro": np.mean(f1_macro), "f1_macro_std": np.std(f1_macro)})
    wandb.log({"f1_micro": np.mean(f1_micro), "f1_micro_std": np.std(f1_micro)})
        
    print("results after rewire: ", np.mean(f1_macro), np.std(f1_macro), np.mean(f1_micro), np.std(f1_micro))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset",             type=str,   default="HGB_DBLP", help="dataset to use. ")
    parser.add_argument("--model",               type=str,   default="SHGN",     help="model to use. ")
    parser.add_argument("--thres_min_deg",       type=float, default=3.,         help="threshhold for minimum degrees. ")
    parser.add_argument("--thres_min_deg_ratio", type=float, default=1.0,        help="threshhold ratio for minimum degrees. ")
    parser.add_argument("--window_size",         type=int,   default=-1,         help="window size used to rewire. ")
    parser.add_argument("--thres_prunning",      type=float, default=0.0,        help="threshhold for edge pruning. ")
    parser.add_argument("--order_neighbors",     type=float, default=2,          help="orde of neighbors to use. ")
    parser.add_argument("--hidden",              type=int,   default=64,         help="hidden size. ")
    parser.add_argument("--epochs",              type=int,   default=400,        help="whether use rewire. ")
    parser.add_argument("--steps",               type=int,   default=1,          help="steps for train loops. ")
    parser.add_argument("--rewire",              type=int,   default=0,          help="whether use rewire. ")
    parser.add_argument("--rewire_whole",        type=int,   default=0,          help="whether use rewire whole. ")
    parser.add_argument("--num-layers",          type=int,   default=2,          help="number of GNN layers. ")
    parser.add_argument("--lr",                  type=float, default=5e-4,       help="learning rate. ")
    parser.add_argument("--weight-decay",        type=float, default=1e-4,       help="weight decay. ")
    parser.add_argument("--slope",               type=float, default=0.05,       help="negative slope. ")
    parser.add_argument("--feats-type",          type=int,   default=3,          help="feature type. ")
    parser.add_argument("--step",                type=int,   default=50,         help="step duration. ")
    parser.add_argument("--use_clf",             type=int,   default=0,          help="step duration. ")
    parser.add_argument("--whole_graph",         type=int,   default=1,          help="whether use whole graph. ")
    parser.add_argument("--use_meta_feat",       type=int,   default=0,          help="whether use the meta features. ")
    
    args = parser.parse_args()
    
    
    if args.model == "GCN":
        args.lr = 1e-3
        args.num_layers = 1
        args.weight_decay = 1e-6
    elif args.model == "GAT":
        args.lr = 1e-3
        args.num_layers = 1
        args.slope = 0.1
    # if args.dataset == "ACM":
    #     if args.model == "GCN":
    #         args.lr = 1e-3
    #         args.weight_decay = 1e-6
    #         args.feats_type = 0
    #     elif args.model == "GAT":
    #         args.feats_type = 2
    #     elif args.model == "SHGN":
    #         args.feats_type = 2
    # elif args.dataset == "DBLP":
    #     if args.model == "GCN":
    #         args.lr = 1e-3
    #         args.weight_decay = 1e-6
    #     elif args.model == "GAT":
    #         pass
    #     elif args.model == "SHGN":
    #         pass
    # elif args.dataset == "HGB_IMDB":
    #     if args.model == "GCN":
    #         args.lr = 1e-3
    #         args.feats_type = 0
    #         args.num_layers = 3
    #     elif args.model == "GAT":
    #         args.lr = 1e-3
    #         args.feats_type = 0
    #         args.num_layers = 4
    #         args.slope = 0.1
    #     elif args.model == "SHGN":
    #         args.feats_type = 0
    #         args.num_layers = 5
    #         args.slope = 0.05

    wandb.init(project="Hetero", entity="gjyspliter", config=args)
    
    config = wandb.config
    
    config = args
    
    main(config)