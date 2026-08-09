import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class FeedforwardBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_type='batch', dropout=0.0, activation='relu'):
        super(FeedforwardBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(in_dim)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(in_dim)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(in_dim)
        else:
            self.norm = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.linear = nn.Linear(in_dim, out_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'celu':
            self.activation = nn.CELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    def forward(self, x):
        if self.norm is not None:
            if len(x.shape) == 2:
                x = self.norm(x)
            else:
                x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.linear(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        return x


class MessagePassingBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mp_type='gcn', norm_type='batch', 
                 dropout=0.0, activation='relu', heads=1):
        super(MessagePassingBlock, self).__init__()
        
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(in_dim)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(in_dim)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(in_dim)
        else:
            self.norm = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.mp_type = mp_type
        if mp_type == 'gcn':
            self.mp_layer = GCNConv(in_dim, out_dim)
        elif mp_type == 'gat':
            self.mp_layer = GATConv(in_dim, out_dim, heads=heads, concat=False)
        elif mp_type == 'mpn':
            self.mp_layer = MPNLayer(in_dim, out_dim)
        elif mp_type == 'tri_mpn':
            self.mp_layer = TriMPNLayer(in_dim, out_dim)
        elif mp_type == 'light_tri_mpn':
            self.mp_layer = LightTriMPNLayer(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown message passing type: {mp_type}")
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'celu':
            self.activation = nn.CELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    def forward(self, x, edge_index, edge_attr=None):
        if self.norm is not None:
            x = self.norm(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.mp_type in ['gcn', 'gat']:
            x = self.mp_layer(x, edge_index)
        else:
            x = self.mp_layer(x, edge_index, edge_attr)
        
        if self.activation is not None:
            x = self.activation(x)
        
        return x


class MPNLayer(MessagePassing):
    """Message Passing Neural Network Layer"""
    def __init__(self, in_dim, out_dim):
        super(MPNLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_dim, out_dim)
        self.lin_edge = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), x.size(1), device=x.device)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        return self.lin_edge(edge_attr) + self.lin(x_j)
    
    def update(self, aggr_out):
        return aggr_out


class TriMPNLayer(MessagePassing):
    """Triplet Message Passing Neural Network Layer"""
    def __init__(self, in_dim, out_dim):
        super(TriMPNLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_dim, out_dim)
        self.lin_edge = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), x.size(1), device=x.device)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Triplet message: includes edge features
        return self.lin_edge(edge_attr) + self.lin(x_j)
    
    def update(self, aggr_out):
        return aggr_out


class LightTriMPNLayer(MessagePassing):
    """Light Triplet Message Passing Neural Network Layer"""
    def __init__(self, in_dim, out_dim):
        super(LightTriMPNLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return self.lin(x_j)
    
    def update(self, aggr_out):
        return aggr_out


class GlobalPoolingBlock(nn.Module):
    def __init__(self, pool_type='mean'):
        super(GlobalPoolingBlock, self).__init__()
        self.pool_type = pool_type
    
    def forward(self, x, batch=None):
        if batch is None:
            if self.pool_type == 'mean':
                return x.mean(dim=0, keepdim=True)
            elif self.pool_type == 'max':
                return x.max(dim=0, keepdim=True)[0]
            elif self.pool_type == 'sum':
                return x.sum(dim=0, keepdim=True)
        else:
            if self.pool_type == 'mean':
                return global_mean_pool(x, batch)
            elif self.pool_type == 'max':
                return global_max_pool(x, batch)
            elif self.pool_type == 'sum':
                return global_add_pool(x, batch)
            else:
                raise ValueError(f"Unknown pool type: {self.pool_type}")


class FusionBlock(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim, fusion_type='concat'):
        super(FusionBlock, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion = nn.Linear(in_dim1 + in_dim2, out_dim)
        elif fusion_type == 'dot':
            assert in_dim1 == in_dim2, "For dot fusion, dimensions must match"
            self.fusion = nn.Linear(in_dim1, out_dim)
        elif fusion_type == 'add':
            assert in_dim1 == in_dim2, "For add fusion, dimensions must match"
            self.fusion = nn.Linear(in_dim1, out_dim)
        elif fusion_type == 'multiply':
            assert in_dim1 == in_dim2, "For multiply fusion, dimensions must match"
            self.fusion = nn.Linear(in_dim1, out_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, x1, x2):
        if self.fusion_type == 'concat':
            x = torch.cat([x1, x2], dim=-1)
        elif self.fusion_type == 'dot':
            x = x1 * x2
        elif self.fusion_type == 'add':
            x = x1 + x2
        elif self.fusion_type == 'multiply':
            x = x1 * x2
        
        return self.fusion(x)

