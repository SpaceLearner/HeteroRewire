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

# from HeteroGraphLearner import ModelHandler as HetModelHandler
# from GraphLearnerRel import ModelHandler
from utils.pytorchtools import EarlyStopping

from gnns import *
 
class GraphModel(nn.Module):
    
    def __init__(self, model_name: str, g, in_dims, num_hidden, num_classes, num_layers, activation, dropout, slope) -> None:
        
        super(GraphModel, self).__init__()
        
        self.model_name = model_name

        print(g)
        
        if model_name == "MLP":
            self.model = MLP(g, in_dims, num_hidden, num_classes, dropout)
        elif model_name == "GCN":
            self.model = GCN(g, in_dims, num_hidden, num_classes, num_layers, activation, dropout)
        elif model_name == "GAT":
            heads = [8] * num_layers + [1]
            self.model = GAT(g, in_dims, num_hidden, num_classes, num_layers, heads, activation, dropout, dropout, slope, True)
        elif model_name == "H2GCN":
            self.model = H2GCN(g, in_dims, num_hidden, num_classes, g.number_of_nodes(), num_layers)
        elif model_name == "LINKX":
            self.model = LINKX(g, in_dims, num_hidden, num_classes, num_layers, g.number_of_nodes())
        elif model_name == "RGCN":
            self.model = RGCN(g, in_dims, num_hidden, num_classes, 30, -1, num_layers, dropout, True)
        elif model_name == "HAN":
            heads = [4] * num_layers + [1]
            self.model = HAN(g, len(g), in_dims[0], 128, num_classes, heads, dropout)
        elif model_name == "HGT":
            self.model = HGT(g, in_dims, num_hidden, num_classes, num_layers, 8)
        elif model_name == "SHGN":
            heads = [8] * num_layers + [1]
            self.model = myGAT(g, 64, 30, in_dims, num_hidden, num_classes, num_layers, heads, activation, dropout, dropout, slope, True, 0.05)
        else:
            raise NotImplementedError
        
    def feat_encode(self, x_list):
        
        return self.model.feat_encode(x_list)
        
    def forward(self, x_list, e_feat):
        
        if self.model_name in ["RGCN", "SHGN"]:
            return self.model(x_list, e_feat)
        elif self.model_name in ["HAN"]:
            return self.model(x_list[0])
        elif self.model_name in ["HGT"]:
            return self.model(0)
        else:
            return self.model(x_list)
        
 
class Trainer(nn.Module):
    
    def __init__(self, model_name, g, in_dims, num_classes, device, config):
        
        super(Trainer, self).__init__()
        
        self.config = config
        
        self.device = device
        
        self.model_name = model_name
        
        self.model     = GraphModel(model_name, g, in_dims, config.hidden, num_classes, config.num_layers, F.elu, 0.5, config.slope)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
    
        self.epochs    = config.epochs
        self.steps     = config.steps
        self.step      = config.step
        
        # # embeddings1 = torch.load("experiments/checkpoints/HGB_"+config.dataset+".emba").to(device)
        # # embeddings2 = torch.load("experiments/checkpoints/HGB_"+config.dataset+".emb").to(device)
        # if config.rewire:
        #     self.embeddings = torch.load("experiments/checkpoints/HGB_"+config.dataset+".embd").to(device)
        #     # self.embeddings = torch.hstack([embeddings1, embeddings2])
            
        #     self.rewire    = ModelHandler(in_dims, num_classes, config.thres_min_deg, config.thres_min_deg_ratio, window_size=[config.window_size, config.window_size], num_epoch=200, device=device, use_clf=config.use_clf)
        #     self.rewire.graph_learner.prepare_x_sim_pre(self.embeddings)
        #     print(self.embeddings.shape)
    
    def forward(self, features_list, e_feat):
        
        return self.model(features_list, e_feat)
    
    def evaluate(self, x_list, e_feat, labels, val_index, test_index, load_model=False, path=None):
        
        with torch.no_grad():
            
            if load_model:
                self.model.load_state_dict(torch.load(path))
            
            self.model.eval()
            logits = self(x_list, e_feat)
            if len(labels.size()) == 1:
                pred          = torch.argmax(logits, dim=1).detach()
                val_loss      = F.cross_entropy(logits[val_index], labels[val_index])
                val_f1_macro  = f1_score(labels[val_index].detach().cpu().numpy(),  pred[val_index].cpu().numpy(),  average="macro")
                test_f1_macro = f1_score(labels[test_index].detach().cpu().numpy(), pred[test_index].cpu().numpy(), average="macro")
                val_f1_micro  = f1_score(labels[val_index].detach().cpu().numpy(),  pred[val_index].cpu().numpy(),  average="micro")
                test_f1_micro = f1_score(labels[test_index].detach().cpu().numpy(), pred[test_index].cpu().numpy(), average="micro")
            else:
                pred     = torch.sigmoid(logits).detach().round()
                val_loss = F.binary_cross_entropy_with_logits(logits[val_index], labels[val_index].float())
                val_f1_macro  = f1_score(labels[val_index].detach().cpu().numpy(),  pred[val_index].cpu().numpy(),  average="macro")
                test_f1_macro = f1_score(labels[test_index].detach().cpu().numpy(), pred[test_index].cpu().numpy(), average="macro")
                val_f1_micro  = f1_score(labels[val_index].detach().cpu().numpy(),  pred[val_index].cpu().numpy(),  average="micro")
                test_f1_micro = f1_score(labels[test_index].detach().cpu().numpy(), pred[test_index].cpu().numpy(), average="micro")
    
        return val_loss, val_f1_macro, val_f1_micro, test_f1_macro, test_f1_micro
    
    def fit_loop(self, x_list, e_feat, labels, train_index, val_index, test_index):
        
        self.model.train()
        self.zero_grad()
        logits = self(x_list, e_feat)
        if len(labels.size()) == 1:
            pred = torch.argmax(logits, dim=1).detach()
            train_f1_macro = f1_score(labels[train_index].detach().cpu().numpy(), pred[train_index].cpu().numpy(), average="macro")
            train_f1_micro = f1_score(labels[train_index].detach().cpu().numpy(), pred[train_index].cpu().numpy(), average="micro")
            
            loss = F.cross_entropy(logits[train_index], labels[train_index])
        else:
            pred = torch.sigmoid(logits).detach().round()
            train_f1_macro = f1_score(labels[train_index].detach().cpu().numpy(), pred[train_index].cpu().numpy(), average="macro")
            train_f1_micro = f1_score(labels[train_index].detach().cpu().numpy(), pred[train_index].cpu().numpy(), average="micro")
            loss = F.binary_cross_entropy_with_logits(logits[train_index], labels[train_index].float())
        
        loss.backward()
        self.optimizer.step()
        
        val_loss, val_f1_macro, val_f1_micro, test_f1_macro, test_f1_micro = self.evaluate(x_list, e_feat, labels, val_index, test_index, False)
    
        return loss, val_loss, train_f1_macro, train_f1_micro, val_f1_macro, val_f1_micro, test_f1_macro, test_f1_micro
    
    def rewire_loop(self, x_list, e_feat, labels, train_index, val_index, test_index):
        
        # features = self.model.feat_encode(x_list).detach()
        num_nodes = sum([len(x) for x in x_list])
        # pos = torch.tensor([0]+[len(x) for x in x_list])
        # pos = torch.cumsum(pos, dim=-1)
        # features = self.embeddings
        features = x_list
        # features_list = [torch.hstack([x_list[i], self.embeddings[pos[i]:pos[i+1]]]) for i in range(len(x_list))]
        # features_list = [x.to(self.device) for x in x_list]
        train_mask = index_to_mask(torch.tensor(train_index), num_nodes)
        val_mask   = index_to_mask(torch.tensor(val_index),   num_nodes)
        test_mask  = index_to_mask(torch.tensor(test_index),  num_nodes)
        
        edge_index = torch.stack(self.model.model.g.edges())
        
        if len(labels.size()) == 1:
            y = torch.hstack([labels, torch.ones(num_nodes-len(labels), device=labels.device)*-1]).long()
        else:
            y = torch.vstack([labels, torch.zeros(num_nodes-len(labels), labels.shape[1], device=labels.device)]).long()
            
        data = Data(x=features[0], x_list=features, num_nodes=num_nodes, e_feat=e_feat, num_target=len(x_list[0]), num_etypes=e_feat.max().item()+1, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(self.device)
        
        data = self.rewire(data, k=8, prunning=False, thres_prunning=self.config.thres_prunning)
        
        num_edges = self.model.model.g.number_of_edges()
        # print(self.model.model.g.device, features.device)
        # print(self.device)
        self.model.model.g.remove_edges(torch.arange(num_edges, device=self.device))
        self.model.model.g.add_edges(data.edge_index[0], data.edge_index[1])
        # self.model.model.g = dgl.to_bidirected(self.model.model.g.cpu()).to(features.device)
        # self.model.model.g = dgl.remove_self_loop(self.model.model.g)
        # self.model.model.g = dgl.add_self_loop(self.model.model.g)
        # self.model.model.g.edges = data.edge_index
        
        # self.rewire.trained = False
        
        return data.e_feat
    
    def fit(self, x_list, e_feat, labels, train_index, val_index, test_index, verbose=True):
        
        path = '../checkpoints/checkpoint_' + str(time.time()) + '.pt'

        early_stopping = EarlyStopping(patience=30, verbose=True, save_path=path)
        # # early_stopping = EarlyStopping(patience=30, verbose=verbose, save_path='../checkpoints/checkpoint.pt')
        # if self.config.rewire:
        #     e_feat = self.rewire_loop(x_list, e_feat, labels, train_index, val_index, test_index)
        
        for i in range(self.epochs):

            for _ in range(self.steps):
                loss, val_loss, \
                train_f1_macro, train_f1_micro, \
                val_f1_macro, val_f1_micro, \
                test_f1_macro, test_f1_micro = self.fit_loop(x_list, e_feat, labels, train_index, val_index, test_index)
            
            # if (i+1) % self.step == 0 and (i+1) <= self.step * 2:
            #     # self.rewire.reset_parameters()
                
            #     early_stopping = EarlyStopping(patience=30, verbose=verbose, save_path=path)
            
            early_stopping(val_loss.item(), self.model)
        
            if verbose:
            
                print('epoch {}, loss is {}, train f1_macro {}, val f1_macro is {}, test f1_macro is {}. '.format(i, loss.item(),
                                                                                                train_f1_macro, val_f1_macro,
                                                                                                test_f1_macro))
            
                print('train f1_micro {}, val f1_micro is {}, test f1_micro is {}. '.format(train_f1_micro, val_f1_micro, test_f1_micro))

            if early_stopping.early_stop:
                print('Early stopping!')
                break
            
        _, _, _, test_f1_macro, test_f1_micro = self.evaluate(x_list, e_feat, labels, val_index, test_index, True, path)

        print('final test result: f1-macro is {}, f1-micro is {}'.format(test_f1_macro, test_f1_micro))

        return test_f1_macro, test_f1_micro
    
class HeteroTrainer(nn.Module):
    
    def __init__(self, model_name, features_list_ori, g, in_dims, num_classes, device, config):
        
        super(HeteroTrainer, self).__init__()
        
        self.config = config
        
        self.device = device
        
        self.model_name = model_name
        
        self.model     = GraphModel(model_name, g, in_dims, config.hidden, num_classes, config.num_layers, F.elu, 0.5, config.slope)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
    
        self.epochs    = config.epochs
        self.steps     = config.steps
        self.step      = config.step
        
        # embeddings = torch.load("experiments/checkpoints/HGB_"+config.dataset+".emb").to(device)
        # nums       = [feature.shape[0] for feature in features_list_ori]
         #embeddings = torch.split(embeddings, nums, dim=0)
        
        self.embeddings_list = features_list_ori # = [torch.hstack([features_list_ori[idx], embeddings[idx]]) for idx in range(len(nums))]
        
        in_dims = [embedding.shape[1] for embedding in self.embeddings_list]

        self.rewire = HetModelHandler(in_dims, num_classes, config.thres_min_deg, config.thres_min_deg_ratio, window_size=[config.window_size, config.window_size], num_epoch=20, device=device, use_clf=config.use_clf)    
    
    def forward(self, features_list, e_feat):
        
        return self.model(features_list, e_feat)
    
    def evaluate(self, x_list, e_feat, labels, val_index, test_index, load_model=False, path=None):
        
        with torch.no_grad():
            
            if load_model:
                self.model.load_state_dict(torch.load(path))
            
            self.model.eval()
            logits = self(x_list, e_feat)
            if len(labels.size()) == 1:
                pred          = torch.argmax(logits, dim=1).detach()
                val_loss      = F.cross_entropy(logits[val_index], labels[val_index])
                val_f1_macro  = f1_score(labels[val_index].detach().cpu().numpy(),  pred[val_index].cpu().numpy(),  average="macro")
                test_f1_macro = f1_score(labels[test_index].detach().cpu().numpy(), pred[test_index].cpu().numpy(), average="macro")
                val_f1_micro  = f1_score(labels[val_index].detach().cpu().numpy(),  pred[val_index].cpu().numpy(),  average="micro")
                test_f1_micro = f1_score(labels[test_index].detach().cpu().numpy(), pred[test_index].cpu().numpy(), average="micro")
            else:
                pred     = torch.sigmoid(logits).detach().round()
                val_loss = F.binary_cross_entropy_with_logits(logits[val_index], labels[val_index].float())
                val_f1_macro  = f1_score(labels[val_index].detach().cpu().numpy(),  pred[val_index].cpu().numpy(),  average="macro")
                test_f1_macro = f1_score(labels[test_index].detach().cpu().numpy(), pred[test_index].cpu().numpy(), average="macro")
                val_f1_micro  = f1_score(labels[val_index].detach().cpu().numpy(),  pred[val_index].cpu().numpy(),  average="micro")
                test_f1_micro = f1_score(labels[test_index].detach().cpu().numpy(), pred[test_index].cpu().numpy(), average="micro")
    
        return val_loss, val_f1_macro, val_f1_micro, test_f1_macro, test_f1_micro
    
    def fit_loop(self, x_list, e_feat, labels, train_index, val_index, test_index):
        
        self.model.train()
        self.zero_grad()
        logits = self(x_list, e_feat)
        if len(labels.size()) == 1:
            pred = torch.argmax(logits, dim=1).detach()
            train_f1_macro = f1_score(labels[train_index].detach().cpu().numpy(), pred[train_index].cpu().numpy(), average="macro")
            train_f1_micro = f1_score(labels[train_index].detach().cpu().numpy(), pred[train_index].cpu().numpy(), average="micro")
            loss = F.cross_entropy(logits[train_index], labels[train_index])
        else:
            pred = torch.sigmoid(logits).detach().round()
            train_f1_macro = f1_score(labels[train_index].detach().cpu().numpy(), pred[train_index].cpu().numpy(), average="macro")
            train_f1_micro = f1_score(labels[train_index].detach().cpu().numpy(), pred[train_index].cpu().numpy(), average="micro")
            loss = F.binary_cross_entropy_with_logits(logits[train_index], labels[train_index].float())
        
        loss.backward()
        self.optimizer.step()
        
        val_loss, val_f1_macro, val_f1_micro, test_f1_macro, test_f1_micro = self.evaluate(x_list, e_feat, labels, val_index, test_index, False)
    
        return loss, val_loss, train_f1_macro, train_f1_micro, val_f1_macro, val_f1_micro, test_f1_macro, test_f1_micro
    
    def rewire_loop(self, x_list, e_feat, labels, train_index, val_index, test_index):
        
        # features = self.model.feat_encode(x_list).detach()
        num_nodes = sum([len(x) for x in x_list])
        # pos = torch.tensor([0]+[len(x) for x in x_list])
        # pos = torch.cumsum(pos, dim=-1)
        features = self.embeddings_list
        # features_list = [torch.hstack([x_list[i], self.embeddings[pos[i]:pos[i+1]]]) for i in range(len(x_list))]
        # features_list = [x.to(self.device) for x in x_list]
        train_mask = index_to_mask(torch.tensor(train_index), num_nodes)
        val_mask   = index_to_mask(torch.tensor(val_index),   num_nodes)
        test_mask  = index_to_mask(torch.tensor(test_index),  num_nodes)
        
        edge_index = torch.stack(self.model.model.g.edges())
        
        if len(labels.size()) == 1:
            y = torch.hstack([labels, torch.ones(num_nodes-len(labels), device=labels.device)*-1]).long()
        else:
            y = torch.vstack([labels, torch.zeros(num_nodes-len(labels), labels.shape[1], device=labels.device)]).long()
        
        data = Data(x=features[0], num_nodes=num_nodes, x_list=features, e_feat=e_feat, num_target=len(x_list[0]), num_etypes=e_feat.max().item()+1, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        data = self.rewire(data.to(self.device), prunning=True, thres_prunning=self.config.thres_prunning)
        
        num_edges = self.model.model.g.number_of_edges()
        self.model.model.g.remove_edges(torch.arange(num_edges, device=features.device))
        self.model.model.g.add_edges(data.edge_index[0], data.edge_index[1])
        # self.model.model.g = dgl.to_bidirected(self.model.model.g.cpu()).to(features.device)
        # self.model.model.g = dgl.remove_self_loop(self.model.model.g)
        # self.model.model.g = dgl.add_self_loop(self.model.model.g)
        # self.model.model.g.edges = data.edge_index
        
        # self.rewire.trained = False
        
        return data.e_feat
    
    def fit(self, x_list, e_feat, labels, train_index, val_index, test_index, verbose=True):
        
        path = '../checkpoints/checkpoint_' + str(time.time()) + '.pt'

        early_stopping = EarlyStopping(patience=30, verbose=True, save_path=path)
        # early_stopping = EarlyStopping(patience=30, verbose=verbose, save_path='../checkpoints/checkpoint.pt')
        if self.config.rewire:
            e_feat = self.rewire_loop(x_list, e_feat, labels, train_index, val_index, test_index)
        
        for i in range(self.epochs):

            for _ in range(self.steps):
                loss, val_loss, \
                train_f1_macro, train_f1_micro, \
                val_f1_macro, val_f1_micro, \
                test_f1_macro, test_f1_micro = self.fit_loop(x_list, e_feat, labels, train_index, val_index, test_index)
            
            # if (i+1) % self.step == 0 and (i+1) <= self.step * 2:
            #     # self.rewire.reset_parameters()
                
            #     early_stopping = EarlyStopping(patience=30, verbose=verbose, save_path=path)
            
            early_stopping(val_loss.item(), self.model)
        
            if verbose:
            
                print('epoch {}, loss is {}, train f1_macro {}, val f1_macro is {}, test f1_macro is {}. '.format(i, loss.item(),
                                                                                                train_f1_macro, val_f1_macro,
                                                                                                test_f1_macro))
            
                print('train f1_micro {}, val f1_micro is {}, test f1_micro is {}. '.format(train_f1_micro, val_f1_micro, test_f1_micro))

            if early_stopping.early_stop:
                print('Early stopping!')
                break
            
        _, _, _, test_f1_macro, test_f1_micro = self.evaluate(x_list, e_feat, labels, val_index, test_index, True, path)

        print('final test result: f1-macro is {}, f1-micro is {}'.format(test_f1_macro, test_f1_micro))

        return test_f1_macro, test_f1_micro
        

