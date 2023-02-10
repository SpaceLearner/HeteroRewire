"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn import GATConv
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

# from sparseVD import LinearSVDO

class PropConv(nn.Module):

    def __init__(self,
                 k,
                 alpha,
                 in_dim,
                 out_dim,
                 num_etypes,
                 edge_drop=0.):
        super(PropConv, self).__init__()
        self._k        = k
        self._alpha    = alpha
        self._in_dim   = in_dim
        self._out_dim  = out_dim
        # self.linear    = nn.Sequential(nn.Linear(self._in_dim, 2 * self._in_dim), nn.GELU(), nn.Linear(2 * self._in_dim, self._out_dim))
        self.linear      = nn.Sequential(nn.Linear(self._in_dim, self._out_dim), nn.GELU(), nn.Linear(self._out_dim, self._out_dim))
        self.edge_embeds = nn.Embedding(num_etypes, self._in_dim)
        self.linear_edge = nn.Sequential(nn.Linear(self._in_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
        self.edge_drop = nn.Dropout(edge_drop)

    def forward(self, graph, feat, e_feat):
        r"""

        Description
        -----------
        Compute APPNP layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)`. :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """

        with graph.local_scope():
            src_norm = th.pow(graph.out_degrees().float().clamp(min=1), -0.5)
            shp = src_norm.shape + (1,) * (feat.dim() - 1)

            src_norm = th.reshape(src_norm, shp).to(feat.device)
            dst_norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = dst_norm.shape + (1,) * (feat.dim() - 1)
            dst_norm = th.reshape(dst_norm, shp).to(feat.device)
            feat_0 = feat
            graph.edata['e'] = self.linear_edge(self.edge_embeds(e_feat))
            for _ in range(self._k):
                # normalization by src node
                feat = feat * src_norm
                graph.ndata['h'] = feat
                graph.edata['w'] = self.edge_drop(
                    th.ones(graph.number_of_edges(), 1).to(feat.device) + graph.edata['e'])
                # graph.apply_edges(fn.u_mul_e('h', 'w', 'm'))

                graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                                fn.sum('m', 'h'))

                feat = graph.ndata.pop('h')
                # normalization by dst node
                feat = feat * dst_norm
                feat = (1 - self._alpha) * feat + self._alpha * feat_0
            return self.linear(feat)

class HeroGATConv(nn.Module):
    
    def __init__(self,
                 in_feats,
                 out_feats,
                 alp_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(HeroGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l  = nn.Parameter(th.FloatTensor(size=(1, num_heads // 2, out_feats)))
        self.attn_r  = nn.Parameter(th.FloatTensor(size=(1, num_heads // 2, out_feats)))
        self.attn_ln = nn.Parameter(th.FloatTensor(size=(1, num_heads // 2, out_feats)))
        self.attn_rn = nn.Parameter(th.FloatTensor(size=(1, num_heads // 2, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        
        self.alp = nn.Sequential(nn.Dropout(), nn.Linear(alp_feats, self._out_feats), nn.ELU(), nn.Linear(self._out_feats, 1), nn.Sigmoid())
        # self.alp_l = nn.Sequential(nn.Dropout(), nn.Linear(alp_feats, 1))
        # self.alp_r = nn.Sequential(nn.Dropout(), nn.Linear(alp_feats, 1))
            
    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_ln, gain=gain)
        nn.init.xavier_normal_(self.attn_rn, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)


    def set_allow_zero_in_degree(self, set_value):
        
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, alpha_hidden, get_attention=False, attn=None):
        
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            
            el  = (feat_src[:, :self._num_heads // 2] * self.attn_l) .sum(dim=-1).unsqueeze(-1)
            er  = (feat_dst[:, :self._num_heads // 2] * self.attn_r) .sum(dim=-1).unsqueeze(-1)
            eln = (feat_src[:, self._num_heads // 2:] * self.attn_ln).sum(dim=-1).unsqueeze(-1)
            ern = (feat_dst[:, self._num_heads // 2:] * self.attn_rn).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src[:, :self._num_heads // 2], 'ftn': feat_src[:, self._num_heads // 2:], 'el': el, 'eln': eln})
            graph.srcdata.update({'ftpf': self.alp(alpha_hidden)})
            graph.dstdata.update({'er': er, 'ern': ern})
            graph.dstdata.update({'ftpt': self.alp(alpha_hidden)})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            graph.apply_edges(fn.u_add_v('eln', 'ern', 'en'))
            graph.apply_edges(fn.u_add_v('ftpf', 'ftpt', 'fte'))
            alpha = torch.sigmoid(graph.edata.pop('fte')[:, :, None])
            
            e  = self.leaky_relu(graph.edata.pop('e'))
            en = self.leaky_relu(graph.edata.pop('en'))
            # compute softmax
            graph.edata['a']  = self.attn_drop(edge_softmax(graph, e)) * alpha
            graph.edata['an'] = self.attn_drop(edge_softmax(graph, -en)) * (1-alpha)# 1.0 / (torch.max(en, torch.ones_like(en, device=en.device) * 1e-12))))
            if attn is not None:
                graph.edata['a']  = graph.edata['a']  * 0.95 + attn[0] * 0.05
                graph.edata['an'] = graph.edata['an'] * 0.95 + attn[1] * 0.05
            # message passing
            # graph.apply_edges(fn.u_mul_e('ft', 'a', 'm'))
            # graph.apply_edges(fn.u_mul_e('ftn', 'an', 'mn'))
            
            # e  = alpha * e
            # en = (1-alpha) * e
            # print(alpha.shape)
            # print(alpha[:, :, None].shape, graph.edata['m'].shape, graph.edata['mn'].shape)
            # graph.edata['m'] = torch.cat([alpha[:, :, None] * graph.edata['m'], (1-alpha[:, :, None]) * graph.edata['mn']], dim=1)
            # graph.edata['m'] = torch.cat([graph.edata['m'], graph.edata['mn']], dim=1) 
            # graph.update_all(fn.copy_e('m', 'm'), fn.sum('m','ft'))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            graph.update_all(fn.u_mul_e('ftn', 'an', 'mn'),
                             fn.sum('mn', 'ftn'))
            rst  = torch.cat([graph.dstdata['ft'], graph.dstdata['ftn']], dim=1)
            # rstn = graph.dstdata['ftn']
            
            # print(alpha_hidden.shape)
            # alpha = self.alp(alpha_hidden)[:, None].detach()# .squeeze()
            # alpha = torch.softmax(alpha / 3, dim=-1).squeeze()
            # alpha = self.attn_drop(alpha)
            # alpha[:] = 0.5
            # print(rst.shape, rstn.shape, alpha.shape)
            # alpha = alpha.permute(0, 2, 1)
            # print(rst.shape, alpha[:, :, None].shape)
            # rst = torch.cat([rst * alpha[:, 0][:, None, None], rstn * alpha[:, 1][:, None, None]], dim=1)
            # rst = torch.cat([rst * alpha, rstn * (1-alpha)], dim=1)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, alpha, (graph.edata['a'].detach(), graph.edata['an'].detach())
            else:
                return rst, alpha
        


"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name

# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('ee'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()

class myGATConvVD(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.,
                 name=None):
        super(myGATConvVD, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        self.name = name
        if isinstance(in_feats, tuple):
            self.fc_src = LinearSVDO(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = LinearSVDO(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = LinearSVDO(
                self._in_src_feats, out_feats * num_heads, bias=False, name=self.name+'_fc')

        self.fc_e = LinearSVDO(edge_feats, edge_feats*num_heads, bias=False, name=self.name+'_fce')
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = LinearSVDO(
                    self._in_dst_feats, num_heads * out_feats, bias=False, name=self.name+'_res')
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        # if hasattr(self, 'fc'):
        #     nn.init.xavier_normal_(self.fc.weight, gain=gain)
        # else:
        #     nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        #     nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        # if isinstance(self.res_fc, nn.Linear):
        #     nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # return feat_dst, None
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat   * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('ee'))
            # e = self.leaky_relu(1 + graph.edata.pop('ee'))
            # e = self.leaky_relu(torch.ones_like(graph.edata.pop('ee')))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()


