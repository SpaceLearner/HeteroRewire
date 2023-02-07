import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import RelGraphConv
from functools import partial

class BaseRGCN(nn.Module):
    def __init__(self, g, in_dims, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.g = g
        self.g.edata["norm"] = torch.zeros(g.number_of_edges(), 1, device=self.g.device)
        edge_index = torch.stack(self.g.edges())
        for i in range(self.g.edata["e_feat"].max().item()+1):
            eid = self.g.edata["e_feat"] == i
            edge_index_i = edge_index[:, eid]
            u, v = edge_index_i
            _, inverse_index, count = torch.unique(
                v, return_inverse=True, return_counts=True)
            degrees = count[inverse_index]
            norm = torch.ones(u.shape[0], device=self.g.device).float() / degrees.float()
            norm = norm.unsqueeze(1)
            self.g.edata["norm"][eid] = norm
        self.in_dims = in_dims
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        self.i2h = self.build_input_layer()
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, features_list, r):
        h = []
        norm = self.g.edata["norm"]
        for i2h, feature in zip(self.i2h, features_list):
            h.append(i2h(feature))
        h = torch.cat(h, 0)
        for layer in self.layers:
            h = layer(self.g, h, r, norm)
        return h

class RGCN(BaseRGCN):

    def build_input_layer(self):
        return nn.ModuleList([nn.Linear(in_dim, self.h_dim, bias=True) for in_dim in self.in_dims])

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, None,
                            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                            dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, None,
                            self.num_bases, activation=None,
                            self_loop=self.use_self_loop)