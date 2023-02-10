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

from .conv import myGATConv
# from HeteroGraphLearner import ModelHandler as HetModelHandler
# from GraphLearnerRel import ModelHandler
from utils.pytorchtools import EarlyStopping
    
    
class MLP(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 dropout):
        super(MLP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(nn.Linear(num_hidden, num_hidden))
        self.layers.append(nn.Linear(num_hidden, num_classes, bias=False))
        self.batchnorm = nn.BatchNorm1d(num_classes, affine=False, track_running_stats=False)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def feat_encode(self, features_list):
        
        h = []
        for i in range(len(features_list)):
            h.append(self.fc_list[i](features_list[i]))
        # for fc, feature in zip(self.fc_list, features_list):
        #     h.append(fc(feature))
        h = torch.cat(h, 0)
        
        return h

    def forward(self, features_list):
        
        with self.g.local_scope():
        
            h = self.feat_encode(features_list)
            
            for i, layer in enumerate(self.layers):
                h = self.dropout(h)
                h = layer(h)
                h = F.elu(h)
                
            # h = self.batchnorm(h)
                
            return h

