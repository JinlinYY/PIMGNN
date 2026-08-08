"""Implement the cgib models gnn baseline module."""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MPNNLayer(MessagePassing):
    """Represent the MPNNLayer baseline component."""
    def __init__(self, in_channels, out_channels, edge_dim=None):
        super(MPNNLayer, self).__init__(aggr='add')
        self.edge_dim = edge_dim
        
        # Baseline workflow step.
        if edge_dim is not None:
            # Baseline workflow step.
            msg_dim = 2 * in_channels + edge_dim
        else:
            # Baseline workflow step.
            msg_dim = 2 * in_channels
        
        self.edge_network = nn.Sequential(
            nn.Linear(msg_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Baseline workflow step.
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Baseline workflow step.
        if edge_attr is not None:
            # Baseline workflow step.
            num_nodes = x.size(0)
            num_edges = edge_index.size(1)
            # Baseline workflow step.
            self_loop_edge_attr = torch.zeros(num_nodes, edge_attr.size(1), 
                                             dtype=edge_attr.dtype, device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, self_loop_edge_attr], dim=0)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:
            msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            msg = torch.cat([x_i, x_j], dim=-1)
        
        return self.edge_network(msg)
    
    def update(self, aggr_out, x):
        return aggr_out + self.lin(x)


class MPNN(nn.Module):
    """Represent the MPNN baseline component."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, edge_dim=None):
        super(MPNN, self).__init__()
        self.num_layers = num_layers
        
        # Baseline workflow step.
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Baseline workflow step.
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, edge_dim)
            for _ in range(num_layers - 1)
        ])
        
        # Configure the output artifacts.
        self.output_layer = MPNNLayer(hidden_dim, output_dim, edge_dim)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_layer(x)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = torch.relu(x)
        
        x = self.output_layer(x, edge_index, edge_attr)
        
        return x


class GINLayer(MessagePassing):
    """Represent the GINLayer baseline component."""
    def __init__(self, in_channels, out_channels, eps=0.0, train_eps=True):
        super(GINLayer, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return x_j
    
    def update(self, aggr_out, x):
        out = self.mlp((1 + self.eps) * x + aggr_out)
        return out


class GIN(nn.Module):
    """Represent the GIN baseline component."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, eps=0.0, train_eps=True):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        
        # Baseline workflow step.
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Baseline workflow step.
        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim, eps, train_eps)
            for _ in range(num_layers - 1)
        ])
        
        # Configure the output artifacts.
        self.output_layer = GINLayer(hidden_dim, output_dim, eps, train_eps)
        
    def forward(self, x, edge_index, batch=None):
        x = self.input_layer(x)
        
        for layer in self.layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
        
        x = self.output_layer(x, edge_index)
        
        return x

