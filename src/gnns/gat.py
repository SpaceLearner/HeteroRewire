import time

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import GATConv, GraphConv
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask

from conv import myGATConv
# from HeteroGraphLearner import ModelHandler as HetModelHandler
# from GraphLearnerRel import ModelHandler
from utils.pytorchtools import EarlyStopping


class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        
    def feat_encode(self, features_list):
        
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        
        return h

    def forward(self, features_list):
        
        with self.g.local_scope():
        
            h = self.feat_encode(features_list)
            
            for l in range(self.num_layers):
                h = self.gat_layers[l](self.g, h).flatten(1)
            # output projection
            logits = self.gat_layers[-1](self.g, h).mean(1)
            return logits