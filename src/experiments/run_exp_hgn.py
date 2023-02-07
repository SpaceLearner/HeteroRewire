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

from models_dgl import Trainer

import argparse

from utils.data import load_data
import copy

def load_dataset(dataname, config=None, device=None):
    
    # seed_everything(seed)
    
    def sp_to_spt(mat):
        coo = mat.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def mat2tensor(mat):
        if type(mat) is np.ndarray:
            return torch.from_numpy(mat).type(torch.FloatTensor)
        return sp_to_spt(mat)


    feats_type = config.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data('/data/guojiayan/HGB1/Homophily/datasets/HGB/', dataname) # load_data('/root/HGB/HGB/NC/benchmark/data', args.dataset) # load_data('/home/gjy/HGB/NC/benchmark/data', args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2 
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            # print(features_list[i].shape)
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    n_types = []

    for idx, feats in enumerate(features_list):
        n_types.append(idx * torch.ones(len(feats), dtype=torch.long))

    n_types = torch.hstack(n_types).to(device)
    
    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])
    

    g = dgl.from_scipy(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    
    return g, in_dims, features_list, e_feat, labels, train_idx, val_idx, test_idx


def run_exp_han(data=[], seed=0, config=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    seed_everything(seed)
    
    g, in_dims, features_list, e_feat, labels, train_index, test_index = data
    
    val_ratio = 0.2
    val_num   = int(len(train_index) * val_ratio)
    
    index = torch.randperm(len(train_index)).numpy()
 
    val_index   = train_index[index[:val_num]]
    train_index = train_index[index[val_num:]]
    
    trainer = Trainer(config.model, copy.deepcopy(g), in_dims, labels.max().item()+1 if len(labels.shape)==1 else labels.shape[1], device, config).to(device)
    
    test_f1_macro, test_f1_micro = trainer.fit(features_list, e_feat, labels, train_index, val_index, test_index, True)

    # test_acc, test_f1_macro, test_f1_micro = model.fit(x_dict, edge_index_dict, labels.to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device), optimizer, target)
    
    return test_f1_macro, test_f1_micro


def main(config):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataname = config.dataset
    
    g, in_dims, features_list, e_feat, labels, train_index, val_index, test_index = load_dataset(dataname, config=config, device=device)
    train_index = np.unique(np.concatenate([train_index, val_index], axis=0))
    data = [g, in_dims, features_list, e_feat, labels, train_index, test_index]
    
    seeds = [213, 4123, 12342, 4213, 5213]
    
    results = []
    
    for idx, seed in enumerate(seeds):
        results.append(run_exp_han(data, seed, config))

    # acc      = []
    f1_macro = []
    f1_micro = []

    for result in results:
       #  acc     .append(result[0])
        f1_macro.append(result[0])
        f1_micro.append(result[1])
        
    # wandb.log({"acc":      np.mean(acc),      "acc_std":      np.std(acc)})
    # wandb.log({"f1_macro": np.mean(f1_macro), "f1_macro_std": np.std(f1_macro)})
    # wandb.log({"f1_micro": np.mean(f1_micro), "f1_micro_std": np.std(f1_micro)})
        
    print("results after rewire: ", np.mean(f1_macro), np.std(f1_macro), np.mean(f1_micro), np.std(f1_micro))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset",             type=str,   default="DBLP",    help="dataset to use. ")
    parser.add_argument("--model",               type=str,   default="GAT",     help="model to use. ")
    parser.add_argument("--thres_min_deg",       type=float, default=16.,        help="threshhold for minimum degrees. ")
    parser.add_argument("--thres_min_deg_ratio", type=float, default=0.5,       help="threshhold ratio for minimum degrees. ")
    parser.add_argument("--window_size",         type=int,   default=10000,     help="window size used to rewire. ")
    parser.add_argument("--thres_prunning",      type=float, default=0.6,       help="threshhold for edge pruning. ")
    parser.add_argument("--order_neighbors",     type=float, default=2,         help="orde of neighbors to use. ")
    parser.add_argument("--hidden",              type=int,   default=64,        help="hidden size. ")
    parser.add_argument("--epochs",              type=int,   default=300,       help="whether use rewire. ")
    parser.add_argument("--steps",               type=int,   default=1,         help="steps for train loops. ")
    parser.add_argument("--rewire",              type=int,   default=0,         help="whether use rewire. ")
    parser.add_argument("--num_layers",          type=int,   default=2,         help="number of GNN layers. ")
    parser.add_argument("--lr",                  type=float, default=5e-4,      help="learning rate. ")
    parser.add_argument("--weight_decay",        type=float, default=1e-4,      help="weight decay. ")
    parser.add_argument("--feats-type",          type=int,   default=3,         help="feature type. ")
    parser.add_argument("--step",                type=int,   default=50,        help="step duration. ")
    
    args = parser.parse_args()
    
    # wandb.init(project="Hetero", entity="gjyspliter", config=args)
    
   #  config = wandb.config
   
    config = args
    
    main(config)
    
    
    