import argparse
import copy

import dgl
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops, add_remaining_self_loops, to_undirected, homophily

from data import load_dataset_pyg
from gnns import HAN
from models_dgl import Trainer

from pytorch_lightning import seed_everything

import numpy as np


def run_exp_han(dataname="HGB_ACM", seed=0, config=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed_everything(seed)

    graph, target, train_mask, val_mask, test_mask, labels, num_labels = load_dataset_pyg(dataname=config.dataset, seed=seed, device=device)

    if dataname[:3] == "HGB":
        # feat_distribution  = torch.load("experiments/checkpoints/" + config.dataset     + ".embd")
       #  feat_distribution  = torch.hstack(feat_distribution)
        features_list      = torch.load("experiments/checkpoints/" + config.dataset[4:] + "_feat.list")
        feat_distribution  = features_list 
    elif dataname[:2] == "FB":
        feat_distribution  = torch.load("experiments/checkpoints/" + config.dataset     + ".embd")
        feat_distribution  = torch.hstack(feat_distribution)
        # print(feat_distribution.shape[0])
        features_list      = [graph[n_type].x.to(device) for n_type in graph.node_types]
        # feat_distribution  = features_list[0]
    else:
        features_list      = [graph[n_type].x.to(device) for n_type in graph.node_types]
        feat_distribution  = features_list[0]

    edge_index_dict_rewire = copy.deepcopy(graph.edge_index_dict)
    for key in list(edge_index_dict_rewire.keys()):
        if "metapath" not in key[1]:
            edge_index_dict_rewire.pop(key)
        else:
            edge_index_dict_rewire[key] = remove_self_loops(edge_index_dict_rewire[key])[0]

    num_nodes = [feat.shape[0] for feat in features_list]

    in_dims = [feat.shape[1] for feat in features_list]

    print(edge_index_dict_rewire.keys())
    for key, item in edge_index_dict_rewire.items():
        print(item.shape)

    g = []
    e_feat = None
    if isinstance(edge_index_dict_rewire, dict):
        edge_index_dict_rewire = list(edge_index_dict_rewire.values())
    for key, _ in enumerate(edge_index_dict_rewire):
        # if "metapath" in key[1]:
        gi = dgl.DGLGraph(num_nodes=num_nodes[0]).to(device)

        edge_index_temp = remove_self_loops(edge_index_dict_rewire[key])[0]
        print("{} node homophily ratio: {}".format(key, homophily(edge_index_temp, labels, method="node")))
        print("{} edge homophily ratio: {}".format(key, homophily(edge_index_temp, labels, method="edge")))
        #  print(gi.number_of_nodes())
        
        edge_index = add_self_loops(to_undirected(edge_index_dict_rewire[key]), num_nodes=num_nodes[0])[0].to(device)
        # print(gi.device, edge_index.device, graph.device)
        gi.add_edges(edge_index[0], edge_index[1])
        # print(gi.number_of_nodes())
        g.append(gi)

    train_index = train_mask.nonzero().squeeze().to(device)
    val_index   = val_mask  .nonzero().squeeze().to(device)
    test_index  = test_mask .nonzero().squeeze().to(device)
    labels      = labels.to(device)

    print(in_dims)

    trainer = Trainer("GCN", g[1], [in_dims[0]], num_labels, device, config).to(device)

    test_f1_macro, test_f1_micro = trainer.fit([features_list[0]], e_feat, labels, train_index, val_index, test_index, True)

    return test_f1_macro, test_f1_micro

def main(args):
    
    dataname = args.dataset
    print(dataname)
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset",             type=str,   default="HGB_DBLP", help="dataset to use. ")
    parser.add_argument("--model",               type=str,   default="HAN",     help="model to use. ")
    parser.add_argument("--thres_min_deg",       type=float, default=3.,         help="threshhold for minimum degrees. ")
    parser.add_argument("--thres_min_deg_ratio", type=float, default=1.0,        help="threshhold ratio for minimum degrees. ")
    parser.add_argument("--window_size",         type=int,   default=-1,         help="window size used to rewire. ")
    parser.add_argument("--thres_prunning",      type=float, default=0.0,        help="threshhold for edge pruning. ", nargs="+")
    parser.add_argument("--order_neighbors",     type=float, default=2,          help="orde of neighbors to use. ")
    parser.add_argument("--hidden",              type=int,   default=64,         help="hidden size. ")
    parser.add_argument("--epochs",              type=int,   default=400,        help="whether use rewire. ")
    parser.add_argument("--steps",               type=int,   default=1,          help="steps for train loops. ")
    parser.add_argument("--rewire",              type=int,   default=0,          help="whether use rewire. ")
    parser.add_argument("--rewire_whole",        type=int,   default=0,          help="whether use rewire whole. ")
    parser.add_argument("--num-layers",          type=int,   default=1,          help="number of GNN layers. ")
    parser.add_argument("--lr",                  type=float, default=5e-4,       help="learning rate. ")
    parser.add_argument("--weight-decay",        type=float, default=1e-4,       help="weight decay. ")
    parser.add_argument("--slope",               type=float, default=0.05,       help="negative slope. ")
    parser.add_argument("--feats-type",          type=int,   default=3,          help="feature type. ")
    parser.add_argument("--step",                type=int,   default=50,         help="step duration. ")
    parser.add_argument("--use_clf",             type=int,   default=1,          help="step duration. ")
    parser.add_argument("--whole_graph",         type=int,   default=1,          help="whether use whole graph. ")
    parser.add_argument("--use_meta_feat",       type=int,   default=0,          help="whether use the meta features. ")

    args = parser.parse_args()

    config = args

    main(config)