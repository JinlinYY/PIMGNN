import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set


# ===== MPNN / Gather / Interaction (merged) =====
class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(MessagePassingLayer, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        
        src_features = x[row]  # [E, node_dim]
        dst_features = x[col]  # [E, node_dim]
        
        message_input = torch.cat([dst_features, src_features, edge_attr], dim=1)
        
        messages = self.message_net(message_input)  # [E, hidden_dim]
        
        num_nodes = x.size(0)
        aggregated_messages = torch.zeros(num_nodes, self.hidden_dim, 
                                         device=x.device, dtype=x.dtype)
        aggregated_messages.index_add_(0, col, messages)
        
        update_input = torch.cat([x, aggregated_messages], dim=1)
        updated_nodes = self.update_net(update_input)
        
        return updated_nodes


class MPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3):
        super(MPNN, self).__init__()
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        h = self.input_proj(x)  # [N, hidden_dim]
        
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_attr)
        
        return h


class GatherLayer(nn.Module):
    def __init__(self, node_dim, hidden_dim, use_set2set=False, processing_steps=3):
        super(GatherLayer, self).__init__()
        self.use_set2set = use_set2set
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        if use_set2set:
            from torch_geometric.nn import Set2Set
            self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps)
            self.output_dim = hidden_dim * 2
        else:
            self.gather_net = nn.Sequential(
                nn.Linear(node_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.output_dim = hidden_dim
    
    def forward(self, x_init, x_mp, batch=None):
        if self.use_set2set:
            if batch is None:
                batch = torch.zeros(x_mp.size(0), dtype=torch.long, device=x_mp.device)
            return self.set2set(x_mp, batch)
        else:
            combined = torch.cat([x_init, x_mp], dim=1)
            return self.gather_net(combined)


class InteractionLayer(nn.Module):
    def __init__(self):
        super(InteractionLayer, self).__init__()
    
    def forward(self, solute_features, solvent_features):
        interaction_map = torch.tanh(
            torch.matmul(solute_features, solvent_features.t())
        )  # [J, K]
        
        solute_weighted = torch.matmul(interaction_map, solvent_features)  # [J, L]
        
        solvent_weighted = torch.matmul(interaction_map.t(), solute_features)  # [K, L]
        
        return interaction_map, solute_weighted, solvent_weighted


# ===== CIGIN model =====



class ReadoutLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_set2set=False, processing_steps=3):
        super(ReadoutLayer, self).__init__()
        self.use_set2set = use_set2set
        
        if use_set2set:
            self.set2set = Set2Set(input_dim, processing_steps=processing_steps)
            self.output_dim = input_dim * 2
        else:
            # Sum pooling
            self.output_dim = input_dim
    
    def forward(self, x, batch=None):
        if self.use_set2set:
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            return self.set2set(x, batch)
        else:
            # Sum pooling
            if batch is None:
                return x.sum(dim=0, keepdim=True)
            else:
                from torch_geometric.nn import global_add_pool
                return global_add_pool(x, batch)


class CIGIN(nn.Module):
    def __init__(self, 
                 node_dim=33,
                 edge_dim=9,
                 hidden_dim=64,
                 num_mp_layers=3,
                 use_set2set=False,
                 set2set_steps=3,
                 use_temperature=True):
        super(CIGIN, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.use_set2set = use_set2set
        self.use_temperature = use_temperature
        
        self.il_mpnn = MPNN(node_dim, edge_dim, hidden_dim, num_mp_layers)
        self.comp2_mpnn = MPNN(node_dim, edge_dim, hidden_dim, num_mp_layers)
        self.comp3_mpnn = MPNN(node_dim, edge_dim, hidden_dim, num_mp_layers)
        
        self.il_gather = GatherLayer(node_dim, hidden_dim, use_set2set, set2set_steps)
        self.comp2_gather = GatherLayer(node_dim, hidden_dim, use_set2set, set2set_steps)
        self.comp3_gather = GatherLayer(node_dim, hidden_dim, use_set2set, set2set_steps)
        
        gather_output_dim = self.il_gather.output_dim
        
        self.interaction_layer = InteractionLayer()
        
        # Generate model predictions.
        self.il_readout = ReadoutLayer(gather_output_dim, hidden_dim, use_set2set, set2set_steps)
        self.comp2_readout = ReadoutLayer(gather_output_dim, hidden_dim, use_set2set, set2set_steps)
        self.comp3_readout = ReadoutLayer(gather_output_dim, hidden_dim, use_set2set, set2set_steps)
        
        readout_output_dim = self.il_readout.output_dim
        
        feature_dim = readout_output_dim * 3
        if use_temperature:
            feature_dim += 1
        # Configure the output artifacts.
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # Configure the output artifacts.
        )
    
    def forward(self, il_data, comp2_data, comp3_data, temperature=None):
        il_mp = self.il_mpnn(
            il_data.x, 
            il_data.edge_index, 
            il_data.edge_attr,
            il_data.batch
        )
        
        comp2_mp = self.comp2_mpnn(
            comp2_data.x,
            comp2_data.edge_index,
            comp2_data.edge_attr,
            comp2_data.batch
        )
        
        comp3_mp = self.comp3_mpnn(
            comp3_data.x,
            comp3_data.edge_index,
            comp3_data.edge_attr,
            comp3_data.batch
        )
        
        # Configure the output artifacts.
        il_features = self.il_gather(il_data.x, il_mp, il_data.batch)
        comp2_features = self.comp2_gather(comp2_data.x, comp2_mp, comp2_data.batch)
        comp3_features = self.comp3_gather(comp3_data.x, comp3_mp, comp3_data.batch)
        
        batch_size = il_data.batch.max().item() + 1 if il_data.batch is not None else 1
        
        il_weighted_list = []
        comp2_weighted_list = []
        comp3_weighted_list = []
        interaction_maps = {'il_comp2': [], 'il_comp3': [], 'comp2_comp3': []}
        
        for i in range(batch_size):
            if batch_size > 1:
                il_mask = (il_data.batch == i)
                comp2_mask = (comp2_data.batch == i)
                comp3_mask = (comp3_data.batch == i)
                il_feat = il_features[il_mask]
                comp2_feat = comp2_features[comp2_mask]
                comp3_feat = comp3_features[comp3_mask]
            else:
                il_feat = il_features
                comp2_feat = comp2_features
                comp3_feat = comp3_features
            
            map_il_comp2, il_weighted_12, comp2_weighted_12 = self.interaction_layer(
                il_feat, comp2_feat
            )
            
            map_il_comp3, il_weighted_13, comp3_weighted_13 = self.interaction_layer(
                il_feat, comp3_feat
            )
            
            map_comp2_comp3, comp2_weighted_23, comp3_weighted_23 = self.interaction_layer(
                comp2_feat, comp3_feat
            )
            
            il_weighted = il_weighted_12 + il_weighted_13
            comp2_weighted = comp2_weighted_12 + comp2_weighted_23
            comp3_weighted = comp3_weighted_13 + comp3_weighted_23
            
            il_weighted_list.append(il_weighted)
            comp2_weighted_list.append(comp2_weighted)
            comp3_weighted_list.append(comp3_weighted)
            
            interaction_maps['il_comp2'].append(map_il_comp2)
            interaction_maps['il_comp3'].append(map_il_comp3)
            interaction_maps['comp2_comp3'].append(map_comp2_comp3)
        
        il_weighted = torch.cat(il_weighted_list, dim=0)
        comp2_weighted = torch.cat(comp2_weighted_list, dim=0)
        comp3_weighted = torch.cat(comp3_weighted_list, dim=0)
        
        # Generate model predictions.
        il_combined = il_features + il_weighted
        comp2_combined = comp2_features + comp2_weighted
        comp3_combined = comp3_features + comp3_weighted
        
        il_graph = self.il_readout(il_combined, il_data.batch)
        comp2_graph = self.comp2_readout(comp2_combined, comp2_data.batch)
        comp3_graph = self.comp3_readout(comp3_combined, comp3_data.batch)
        
        combined_features = torch.cat([il_graph, comp2_graph, comp3_graph], dim=1)
        
        if self.use_temperature and temperature is not None:
            if temperature.dim() == 1:
                temperature = temperature.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
            combined_features = torch.cat([combined_features, temperature], dim=1)
        
        # Generate model predictions.
        prediction = self.predictor(combined_features)
        
        sample_interaction_maps = {
            'il_comp2': interaction_maps['il_comp2'][0] if len(interaction_maps['il_comp2']) > 0 else None,
            'il_comp3': interaction_maps['il_comp3'][0] if len(interaction_maps['il_comp3']) > 0 else None,
            'comp2_comp3': interaction_maps['comp2_comp3'][0] if len(interaction_maps['comp2_comp3']) > 0 else None
        }
        
        return prediction, sample_interaction_maps

