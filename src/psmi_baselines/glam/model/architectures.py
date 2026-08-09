import torch
import torch.nn as nn
from .blocks import FeedforwardBlock, MessagePassingBlock, GlobalPoolingBlock, FusionBlock


class SingleGraphArchitecture(nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim, num_mp_layers=3,
                 mp_type='gcn', norm_type='batch', dropout=0.0, 
                 activation='relu', pool_type='mean'):
        super(SingleGraphArchitecture, self).__init__()
        
        self.input_ff = FeedforwardBlock(
            node_dim, hidden_dim, norm_type, dropout, activation
        )
        
        # Message-passing blocks
        self.mp_layers = nn.ModuleList([
            MessagePassingBlock(
                hidden_dim, hidden_dim, mp_type, norm_type, 
                dropout, activation
            ) for _ in range(num_mp_layers)
        ])
        
        # Global pooling block
        self.global_pool = GlobalPoolingBlock(pool_type)
        
        # Configure the output artifacts.
        self.output_ff = FeedforwardBlock(
            hidden_dim, out_dim, norm_type, dropout, activation
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_ff(x)
        
        for mp_layer in self.mp_layers:
            x = mp_layer(x, edge_index, edge_attr)
        
        x = self.global_pool(x, batch)
        
        # Configure the output artifacts.
        x = self.output_ff(x)
        
        return x


class PairGraphArchitecture(nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim, num_mp_layers=3,
                 mp_type='gcn', norm_type='batch', dropout=0.0,
                 activation='relu', pool_type='mean', fusion_type='concat'):
        super(PairGraphArchitecture, self).__init__()
        
        self.input_ff1 = FeedforwardBlock(
            node_dim, hidden_dim, norm_type, dropout, activation
        )
        self.input_ff2 = FeedforwardBlock(
            node_dim, hidden_dim, norm_type, dropout, activation
        )
        
        self.mp_layers1 = nn.ModuleList([
            MessagePassingBlock(
                hidden_dim, hidden_dim, mp_type, norm_type,
                dropout, activation
            ) for _ in range(num_mp_layers)
        ])
        self.mp_layers2 = nn.ModuleList([
            MessagePassingBlock(
                hidden_dim, hidden_dim, mp_type, norm_type,
                dropout, activation
            ) for _ in range(num_mp_layers)
        ])
        
        # Global pooling blocks
        self.global_pool1 = GlobalPoolingBlock(pool_type)
        self.global_pool2 = GlobalPoolingBlock(pool_type)
        
        # Fusion block
        self.fusion = FusionBlock(
            hidden_dim, hidden_dim, hidden_dim, fusion_type
        )
        
        # Configure the output artifacts.
        self.output_ff = FeedforwardBlock(
            hidden_dim, out_dim, norm_type, dropout, activation
        )
    
    def forward(self, x1, edge_index1, x2, edge_index2, 
                edge_attr1=None, edge_attr2=None, 
                batch1=None, batch2=None):
        x1 = self.input_ff1(x1)
        x2 = self.input_ff2(x2)
        
        for mp_layer in self.mp_layers1:
            x1 = mp_layer(x1, edge_index1, edge_attr1)
        for mp_layer in self.mp_layers2:
            x2 = mp_layer(x2, edge_index2, edge_attr2)
        
        x1 = self.global_pool1(x1, batch1)
        x2 = self.global_pool2(x2, batch2)
        
        x = self.fusion(x1, x2)
        
        # Configure the output artifacts.
        x = self.output_ff(x)
        
        return x


class TripleGraphArchitecture(nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim, num_mp_layers=3,
                 mp_type='gcn', norm_type='batch', dropout=0.0,
                 activation='relu', pool_type='mean', fusion_type='concat'):
        super(TripleGraphArchitecture, self).__init__()
        
        self.input_ff1 = FeedforwardBlock(
            node_dim, hidden_dim, norm_type, dropout, activation
        )
        self.input_ff2 = FeedforwardBlock(
            node_dim, hidden_dim, norm_type, dropout, activation
        )
        self.input_ff3 = FeedforwardBlock(
            node_dim, hidden_dim, norm_type, dropout, activation
        )
        
        self.mp_layers1 = nn.ModuleList([
            MessagePassingBlock(
                hidden_dim, hidden_dim, mp_type, norm_type,
                dropout, activation
            ) for _ in range(num_mp_layers)
        ])
        self.mp_layers2 = nn.ModuleList([
            MessagePassingBlock(
                hidden_dim, hidden_dim, mp_type, norm_type,
                dropout, activation
            ) for _ in range(num_mp_layers)
        ])
        self.mp_layers3 = nn.ModuleList([
            MessagePassingBlock(
                hidden_dim, hidden_dim, mp_type, norm_type,
                dropout, activation
            ) for _ in range(num_mp_layers)
        ])
        
        # Global pooling blocks
        self.global_pool1 = GlobalPoolingBlock(pool_type)
        self.global_pool2 = GlobalPoolingBlock(pool_type)
        self.global_pool3 = GlobalPoolingBlock(pool_type)
        
        # Triple Fusion block
        self.fusion = TripleFusionBlock(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, fusion_type
        )
        
        self.temp_embedding = nn.Linear(1, hidden_dim)
        self.use_temperature = True
        # Configure the output artifacts.
        # Configure the output artifacts.
        fusion_out_dim = hidden_dim
        if self.use_temperature:
            fusion_out_dim += hidden_dim
        
        self.output_ff = FeedforwardBlock(
            fusion_out_dim, out_dim, norm_type, dropout, activation
        )
    
    def forward(self, x1, edge_index1, x2, edge_index2, x3, edge_index3,
                edge_attr1=None, edge_attr2=None, edge_attr3=None,
                batch1=None, batch2=None, batch3=None, temperature=None):
        x1 = self.input_ff1(x1)
        x2 = self.input_ff2(x2)
        x3 = self.input_ff3(x3)
        
        for mp_layer in self.mp_layers1:
            x1 = mp_layer(x1, edge_index1, edge_attr1)
        for mp_layer in self.mp_layers2:
            x2 = mp_layer(x2, edge_index2, edge_attr2)
        for mp_layer in self.mp_layers3:
            x3 = mp_layer(x3, edge_index3, edge_attr3)
        
        x1 = self.global_pool1(x1, batch1)
        x2 = self.global_pool2(x2, batch2)
        x3 = self.global_pool3(x3, batch3)
        
        x = self.fusion(x1, x2, x3)
        
        if temperature is not None and self.use_temperature:
            if temperature.dim() == 1:
                temperature = temperature.unsqueeze(-1)
            temp_feat = self.temp_embedding(temperature)
            x = torch.cat([x, temp_feat], dim=-1)
        
        # Configure the output artifacts.
        x = self.output_ff(x)
        
        return x


class TripleFusionBlock(nn.Module):
    def __init__(self, in_dim1, in_dim2, in_dim3, out_dim, fusion_type='concat'):
        super(TripleFusionBlock, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion = nn.Linear(in_dim1 + in_dim2 + in_dim3, out_dim)
        elif fusion_type == 'add':
            assert in_dim1 == in_dim2 == in_dim3, "For add fusion, dimensions must match"
            self.fusion = nn.Linear(in_dim1, out_dim)
        elif fusion_type == 'multiply':
            assert in_dim1 == in_dim2 == in_dim3, "For multiply fusion, dimensions must match"
            self.fusion = nn.Linear(in_dim1, out_dim)
        elif fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(in_dim1, num_heads=4, batch_first=True)
            self.fusion = nn.Linear(in_dim1, out_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, x1, x2, x3):
        if self.fusion_type == 'concat':
            x = torch.cat([x1, x2, x3], dim=-1)
        elif self.fusion_type == 'add':
            x = x1 + x2 + x3
        elif self.fusion_type == 'multiply':
            x = x1 * x2 * x3
        elif self.fusion_type == 'attention':
            x_stack = torch.stack([x1, x2, x3], dim=1)  # [batch, 3, dim]
            x_attn, _ = self.attention(x_stack, x_stack, x_stack)
            x = x_attn.mean(dim=1)
        return self.fusion(x)

