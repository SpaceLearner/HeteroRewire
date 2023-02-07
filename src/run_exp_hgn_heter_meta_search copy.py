import sys
from unittest import findTestCases

sys.path.append("../src")
import argparse
import torch
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.datasets import DBLP, IMDB
from torch_geometric.utils import remove_self_loops, remove_isolated_nodes, homophily, add_self_loops, to_undirected, index_to_mask
from torch_geometric.transforms import AddMetaPaths

from pytorch_lightning import seed_everything

from dataset import *
from data import load_dataset_pyg

from models_dgl import Trainer

from HeteroGraphLearner import ModelHandler

import argparse
import copy
from torch_sparse import coalesce

import networkx as nx

import wandb

import dgl

import time


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
    # for key in list(edge_index_dict.keys()):
    #     if key[0] == target and key[0] == key[2]:
    #         edge_index_dict.pop(key)
    
    edge_index_dict_rewire = copy.deepcopy(graph.edge_index_dict)
    for key in list(edge_index_dict_rewire.keys()):
        if "metapath" not in key[1]:
            edge_index_dict_rewire.pop(key)
    
    print(edge_index_dict_rewire.keys())
    
    # if config.dataset in ["Actor", "Liar", "Fake_politifact", "Fake_gossipcop", "FD_Amazon"]:
        #         feat_distribution = feat_distribution.to_dense()
    
    if config.rewire_whole:
        
        for key, edge_index_rewire in edge_index_dict_rewire.items(): 
            print("metapath: {}, homophily ratio is {}".format(key, homophily(edge_index_rewire, y=graph[target].y)))
        
        in_sizes       = [feature.shape[1] for feature in features_list]
        rewirer        = ModelHandler(in_sizes, len(edge_index_dict_rewire), num_labels, device=device, window_size=[config.window_size, config.window_size], num_epoch=200, num_epoch_finetune=50, shuffle=[True, True], thres_min_deg=config.thres_min_deg, thres_min_deg_ratio=config.thres_min_deg_ratio, use_clf=config.use_clf)
        edge_index     = torch.hstack(list(edge_index_dict.values()))
        edge_index     = add_self_loops(to_undirected(edge_index))[0]
        data0          = Data(edge_index=edge_index, xs=features_list, num_targets=features_list[0].shape[0], y=graph[target].y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        thres_prunning = config.thres_prunning
        
        if not os.path.exists(config.saved_path):
            edge_index_dict_rewire = rewirer(data0, edge_index_dict_rewire, prunning=True, thres_prunning=thres_prunning, save_path=config.saved_path)
        else:
            edge_index_dict_rewire = rewirer(data0, edge_index_dict_rewire, prunning=True, thres_prunning=thres_prunning, load_path=config.saved_path)
        
        cnt = 0
        for key, edge_index_rewire in enumerate(edge_index_dict_rewire): 
            print("metapath: {}, after rewire homophily ratio is {}".format(key, homophily(edge_index_dict_rewire[key], y=graph[target].y)))
            edge_index_dict[(target, "metapath_"+str(cnt), target)] = to_undirected(edge_index_dict_rewire[key]).to(device)
    
        for key in list(edge_index_dict.keys()):
            if key[0] == target and key[0] == key[2] and "metapath" not in key[1]:
                edge_index_dict.pop(key)
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
        
    print("results after rewire: ", np.mean(f1_macro), np.std(f1_macro), np.mean(f1_micro), np.std(f1_micro))

    wandb.log({"f1_macro": np.mean(f1_macro), "f1_macro_std": np.std(f1_macro)})
    wandb.log({"f1_micro": np.mean(f1_micro), "f1_micro_std": np.std(f1_micro)})
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset",             type=str,   default="HGB_DBLP", help="dataset to use. ")
    parser.add_argument("--model",               type=str,   default="SHGN",     help="model to use. ")
    parser.add_argument("--thres_min_deg",       type=float, default=8.,         help="threshhold for minimum degrees. ")
    parser.add_argument("--thres_min_deg_ratio", type=float, default=1.0,        help="threshhold ratio for minimum degrees. ")
    parser.add_argument("--window_size",         type=int,   default=-1,         help="window size used to rewire. ")
    parser.add_argument("--thres_prunning",      type=float, default=0.0,        help="threshhold for edge pruning. ", nargs="+")
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
        args.num_layers = 2
        args.weight_decay = 1e-6
    elif args.model == "H2GCN":
        args.lr = 1e-3
        args.num_layers = 2
        args.weight_decay = 1e-6
    elif args.model == "RGCN":
        args.lr = 1e-2
        args.num_layers = 4
        args.hidden = 16
    elif args.model == "GAT":
        args.lr = 1e-3
        args.num_layers = 2
        args.slope = 0.1
        
    if args.dataset == "Actor":
        args.thres_prunning = [0.3, 0.9, 0.9]
    elif args.dataset[:2] == "FB":
        args.thres_prunning = [-1.0, 0.9, 0.9]
    elif args.dataset == "Liar":
        args.thres_prunning = [0.9, 0.9, 0.9]
        
    args.saved_path = "../saved_graph/graph." + str(time.time())
        
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