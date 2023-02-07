import os.path as osp
import sys
sys.path.append("../../src")
sys.path.append("../")

import torch
import torch.nn as nn

from torch_geometric.datasets import AMiner, HGBDataset
from torch_geometric.nn import MetaPath2Vec, GAE, VGAE, GCNConv, HeteroLinear
from torch_geometric.transforms import AddMetaPaths
from datasets import *

import argparse

from tqdm import tqdm
from utils.data import load_data

class HGCNEncoder(torch.nn.Module):
    def __init__(self, in_dims, in_channels, out_channels):
        super().__init__()
        self.proj  = nn.ModuleList()
        for i in range(len(in_dims)):
            self.proj.append(nn.Linear(in_dims[i], in_channels))
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x_list, edge_index):
        x = torch.vstack([self.proj[i](x) for i, x in enumerate(x_list)])
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


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
    # for feature in features_list:
    #     print(feature.shape)
    # print(dl.pos)
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
    val_idx   = train_val_test_idx['val_idx']
    val_idx   = np.sort(val_idx)
    test_idx  = train_val_test_idx['test_idx']
    test_idx  = np.sort(test_idx)

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

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataname = args.dataset

    g, in_dims, features_list, e_feat, labels, train_idx, val_idx, test_idx = load_dataset(dataname, device=device, config=args)
       
    if args.dataset == "HGB_ACM":
        targets = ["paper", "author", "subject", "term"] 
    elif args.dataset == "HGB_DBLP":
        targets = ["author", "paper", "term", "venue"]
    elif args.dataset == "HGB_IMDB":
        targets = ["movie", "director", "actor", "keyword"]
    else:
        targets = ["book", "film", "location", "music", "person", "sport", "organization", "business"]
   

    model   = GAE(HGCNEncoder(in_dims, in_channels=128, out_channels=128))
    
    # if not args.variational and not args.linear:
    #     model = GAE(GCNEncoder(in_channels, out_channels))
    # elif not args.variational and args.linear:
    #     model = GAE(LinearEncoder(in_channels, out_channels))
    # elif args.variational and not args.linear:
    #     model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    # elif args.variational and args.linear:
    #     model = VGAE(VariationalLinearEncoder(in_channels, out_channels))
    g = g.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    edge_index = torch.stack(g.edges())
    # print(edge_index.shape)
    # pos_edge_label_index = torch.arange(edge_index.shape[1])

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(features_list, edge_index)
        # print(z.shape, pos_edge_label_index.shape)
        loss = model.recon_loss(z, edge_index)
        if args.variational:
            loss = loss + (1 / g.number_of_nodes()) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)


    # @torch.no_grad()
    # def test(data):
    #     model.eval()
    #     z = model.encode(data.x, data.edge_index)
    #     return model.test(z, pos_edge_label_index, data.neg_edge_label_index)


    for epoch in range(1, 1001):
        loss = train()
        if epoch % 50 == 0:
            # val_acc, test_acc = test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
                # f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        
    print(model)
    zs = []
    with torch.no_grad():
        model.eval()
        # for idx, target1 in enumerate(targets):
        zs = model.encode(features_list, edge_index)
        torch.save(zs, "checkpoints/"+dataname+".emba")
            
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="ACM", help="dataset to use. ")
    parser.add_argument("--feats-type",  default=0,     help="feats type. ")
    parser.add_argument("--variational", default=0,     help="wether use vational method. ")
    # parser.add_argument("--metapath", default="HGB_ACM", help="dataset to use. ")
    # parser.add_argument("--target",   default="au", help="dataset to use. ")
    
    args = parser.parse_args()
    
    main(args)
