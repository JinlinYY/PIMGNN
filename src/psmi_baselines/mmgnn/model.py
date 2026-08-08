# -*- coding: utf-8 -*-
"""Implement the mmgnn model baseline module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, Set2Set
from torch_geometric.data import Batch, Data
from torch_geometric.utils import add_self_loops

from .graph_builder import MoleculeGraphBuilder, ATOM_FEAT_DIM, BOND_FEAT_DIM


class IntramolecularMessagePassing(MessagePassing):
    """Represent the IntramolecularMessagePassing baseline component."""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__(aggr='add', flow='source_to_target')
        self.node_feat_dim = node_dim  # Baseline workflow step.
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Baseline workflow step.
        self.edge_net = nn.Sequential(
            nn.Linear(self.node_feat_dim * 2 + edge_dim + 6, hidden_dim),  # +6 for global features
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, edge_dim),
        )
        
        # Baseline workflow step.
        self.node_net = nn.Sequential(
            nn.Linear(self.node_feat_dim + hidden_dim + 6, hidden_dim),  # +6 for global features
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.node_feat_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.eps = 0.0001
    
    def forward(self, x, edge_index, edge_attr, u, batch=None):
        """Run the forward baseline operation."""
        # Baseline workflow step.
        # Baseline workflow step.
        num_edges_before = edge_index.size(1)
        
        # Baseline workflow step.
        if edge_attr.size(0) != num_edges_before:
            # Baseline workflow step.
            if edge_attr.size(0) > num_edges_before:
                edge_attr = edge_attr[:num_edges_before]
            # Baseline workflow step.
            else:
                padding = torch.zeros(
                    num_edges_before - edge_attr.size(0), edge_attr.size(1),
                    dtype=edge_attr.dtype, device=edge_attr.device
                )
                edge_attr = torch.cat([edge_attr, padding], dim=0)
        
        # Baseline workflow step.
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        
        # Baseline workflow step.
        num_edges_after = edge_index.size(1)
        if edge_attr.size(0) != num_edges_after:
            # Baseline workflow step.
            num_self_loops = num_edges_after - num_edges_before
            if num_self_loops > 0:
                self_loop_attr = torch.zeros(
                    num_self_loops, edge_attr.size(1),
                    dtype=edge_attr.dtype, device=edge_attr.device
                )
                edge_attr = torch.cat([edge_attr[:num_edges_before], self_loop_attr], dim=0)
            else:
                edge_attr = edge_attr[:num_edges_after]
        
        # Baseline workflow step.
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        
        # Baseline workflow step.
        # Baseline workflow step.
        if u.size(1) != 6:
            raise ValueError(f"u shape Incorrect : {u.shape}, expected number Two dimensions is 6")
        
        if batch is not None:
            u_expanded = u[batch[row]]  # (E, 6)
        else:
            u_expanded = u.expand(edge_index.size(1), -1)  # Baseline workflow step.
        
        # Evaluate the validation subset.
        expected_dim = x_i.size(1) * 2 + edge_attr.size(1) + u_expanded.size(1)
        actual_dim = x_i.size(1) + x_j.size(1) + edge_attr.size(1) + u_expanded.size(1)
        if actual_dim != expected_dim:
            raise ValueError(f"edge_input dimension mismatch : x_i={x_i.size(1)}, x_j={x_j.size(1)}, "
                           f"edge_attr={edge_attr.size(1)}, u_expanded={u_expanded.size(1)}, "
                           f" expected Total dimension ={self.node_feat_dim * 2 + self.edge_dim + 6}, "
                           f" Actual Total dimension ={actual_dim}")
        
        edge_input = torch.cat([x_i, x_j, edge_attr, u_expanded], dim=-1)
        edge_attr_updated = edge_attr + self.edge_net(edge_input)
        
        # Baseline workflow step.
        edge_weights = torch.sigmoid(edge_attr_updated.sum(dim=-1, keepdim=True))
        edge_weights = edge_weights / (edge_weights.sum() + self.eps)
        
        # Baseline workflow step.
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr_updated, 
                            edge_weights=edge_weights, u=u, batch=batch)
        
        return out, edge_attr_updated
    
    def message(self, x_j, edge_attr, edge_weights):
        # Baseline workflow step.
        return edge_weights * x_j
    
    def update(self, aggr_out, x, u, batch):
        # Baseline workflow step.
        if batch is not None:
            u_expanded = u[batch]
        else:
            u_expanded = u.expand(x.size(0), -1)
        
        # Baseline workflow step.
        node_input = torch.cat([x, aggr_out, u_expanded], dim=-1)
        x_updated = x + self.node_net(node_input)
        return self.dropout(x_updated)


class IntermolecularMessagePassing(MessagePassing):
    """Represent the IntermolecularMessagePassing baseline component."""
    
    def __init__(self, node_dim: int, hidden_dim: int, beta: float = 0.2, dropout: float = 0.15):
        super().__init__(aggr='add', flow='source_to_target')
        self.node_feat_dim = node_dim  # Baseline workflow step.
        self.hidden_dim = hidden_dim
        self.beta = beta  # Baseline workflow step.
        
        # Baseline workflow step.
        self.similarity_net = nn.Sequential(
            nn.Linear(self.node_feat_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Baseline workflow step.
        self.node_net = nn.Sequential(
            nn.Linear(self.node_feat_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.node_feat_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.eps = 0.0001
    
    def forward(self, x, edge_index, molecule_ranges):
        """Run the forward baseline operation."""
        # Baseline workflow step.
        if edge_index.size(1) == 0:
            # Baseline workflow step.
            return x
        
        # Baseline workflow step.
        num_nodes = x.size(0)
        max_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
        min_idx = edge_index.min().item() if edge_index.numel() > 0 else -1
        
        if max_idx >= num_nodes or min_idx < 0:
            # Baseline workflow step.
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) &\
                         (edge_index[0] >= 0) & (edge_index[1] >= 0)
            edge_index = edge_index[:, valid_mask]
            
            if edge_index.size(1) == 0:
                return x
        
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        
        # Baseline workflow step.
        similarity = self.similarity_net(torch.cat([x_i, x_j], dim=-1))
        similarity = 1.0 / (torch.abs(x_i - x_j).sum(dim=-1, keepdim=True) + self.eps)
        
        # Baseline workflow step.
        # Baseline workflow step.
        out = self.propagate(edge_index, x=x, similarity=similarity, size=(num_nodes, num_nodes))
        
        # Baseline workflow step.
        if out.size(0) != num_nodes:
            # Baseline workflow step.
            out_padded = torch.zeros(num_nodes, out.size(1), dtype=out.dtype, device=out.device)
            if out.size(0) <= num_nodes:
                out_padded[:out.size(0)] = out
            else:
                out_padded = out[:num_nodes]
            out = out_padded
        
        # Baseline workflow step.
        x_updated = (1 - self.beta) * x + self.beta * self.node_net(torch.cat([x, out], dim=-1))
        
        return self.dropout(x_updated)
    
    def message(self, x_j, similarity):
        # Baseline workflow step.
        return similarity * x_j


class Explainer(nn.Module):
    """Represent the Explainer baseline component."""
    
    def __init__(self, node_dim: int, edge_dim: int, method: str = 'local_mask'):
        """
        method: 'local_mask', 'global_mask', 'gradient'
        """
        super().__init__()
        self.method = method
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        if method == 'local_mask':
            # Baseline workflow step.
            self.node_mask = nn.Parameter(torch.ones(1))  # Baseline workflow step.
            self.edge_mask = nn.Parameter(torch.ones(1))
        
        elif method == 'global_mask':
            # Baseline workflow step.
            self.node_mask_net = nn.Sequential(
                nn.Linear(node_dim, 1),
                nn.Sigmoid()
            )
            self.edge_mask_net = nn.Sequential(
                nn.Linear(edge_dim, 1),
                nn.Sigmoid()
            )
        
        # Configure experiment parameters.
    
    def forward(self, node_features, edge_features=None, return_masks=False):
        """Run the forward baseline operation."""
        if self.method == 'local_mask':
            node_mask = torch.sigmoid(self.node_mask)
            masked_nodes = node_features * node_mask
            
            if edge_features is not None:
                edge_mask = torch.sigmoid(self.edge_mask)
                masked_edges = edge_features * edge_mask
            else:
                masked_edges = None
                edge_mask = None
        
        elif self.method == 'global_mask':
            node_mask = self.node_mask_net(node_features)
            masked_nodes = node_features * node_mask
            
            if edge_features is not None:
                edge_mask = self.edge_mask_net(edge_features)
                masked_edges = edge_features * edge_mask
            else:
                masked_edges = None
                edge_mask = None
        
        else:  # Run the training step.
            masked_nodes = node_features
            masked_edges = edge_features
            node_mask = None
            edge_mask = None
        
        if return_masks:
            return masked_nodes, masked_edges, node_mask, edge_mask
        return masked_nodes, masked_edges


class MMGNN(nn.Module):
    """Represent the MMGNN baseline component."""
    
    def __init__(
        self,
        node_dim: int = ATOM_FEAT_DIM,
        edge_dim: int = BOND_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        set2set_steps: int = 3,  # Run the training step.
        post_explain_layers: int = 2,  # Run the training step.
        beta: float = 0.2,
        explainer_method: str = 'local_mask',
        dropout: float = 0.15,
        output_dim: int = 6,  # Configure the output artifacts.
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.beta = beta
        
        # Baseline workflow step.
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.global_proj = nn.Linear(6, hidden_dim)  # Baseline workflow step.
        
        # Baseline workflow step.
        self.intra_layers = nn.ModuleList([
            IntramolecularMessagePassing(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Baseline workflow step.
        self.inter_layer = IntermolecularMessagePassing(hidden_dim, hidden_dim, beta, dropout)
        
        # Baseline workflow step.
        self.explainer = Explainer(hidden_dim, hidden_dim, explainer_method)
        
        # Baseline workflow step.
        self.post_explain_layers = nn.ModuleList([
            IntramolecularMessagePassing(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(post_explain_layers)
        ])
        
        # Baseline workflow step.
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)
        
        # Generate model predictions.
        # Configure the output artifacts.
        # Baseline workflow step.
        predictor_input_dim = hidden_dim * 2 * 3 + 2  # Baseline workflow step.
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self._phase_softmax = lambda y: torch.cat([
            F.softmax(y[:, :3], dim=-1),
            F.softmax(y[:, 3:], dim=-1)
        ], dim=-1)
    
    def forward(self, batch_data, T_norm, t, return_explanation=False):
        """Run the forward baseline operation."""
        x = batch_data.x  # (N, node_dim)
        edge_index = batch_data.edge_index  # (2, E)
        edge_attr = batch_data.edge_attr  # (E, edge_dim)
        batch = batch_data.batch  # Baseline workflow step.
        
        # Baseline workflow step.
        # Baseline workflow step.
        if hasattr(batch_data, 'molecule_ranges') and batch_data.molecule_ranges is not None:
            # Baseline workflow step.
            if batch is None or batch.max().item() == 0:
                molecule_ranges = batch_data.molecule_ranges if isinstance(batch_data.molecule_ranges, list) else None
            else:
                # Baseline workflow step.
                # Baseline workflow step.
                molecule_ranges = batch_data.molecule_ranges[0] if isinstance(batch_data.molecule_ranges, list) else None
        else:
            molecule_ranges = None
        
        # Baseline workflow step.
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Baseline workflow step.
        # Baseline workflow step.
        # Baseline workflow step.
        # Baseline workflow step.
        u = batch_data.global_features
        
        # Baseline workflow step.
        if u.dim() == 1:
            # Baseline workflow step.
            # Baseline workflow step.
            if batch is not None:
                batch_size = batch.max().item() + 1
                # Baseline workflow step.
                if u.size(0) == batch_size * 3 * 6:
                    u = u.view(batch_size * 3, 6)
                elif u.size(0) == batch_size * 18:
                    u = u.view(batch_size, 18)
                else:
                    # Baseline workflow step.
                    if u.size(0) % 6 == 0:
                        u = u.view(-1, 6)
                    else:
                        raise ValueError(f" unable to process 1D global_features shape : {u.shape}")
            else:
                # Baseline workflow step.
                if u.size(0) % 6 == 0:
                    u = u.view(-1, 6)
                else:
                    raise ValueError(f" unable to process 1D global_features shape : {u.shape}")
        elif u.dim() > 2:
            # Baseline workflow step.
            u = u.view(-1, u.size(-1))
        
        # Baseline workflow step.
        if batch is not None:
            batch_size = batch.max().item() + 1
            
            # Baseline workflow step.
            if u.size(0) == batch_size * 3 and u.size(1) == 6:
                # Baseline workflow step.
                u_list = []
                for b in range(batch_size):
                    start_idx = b * 3
                    end_idx = (b + 1) * 3
                    u_batch = u[start_idx:end_idx]  # (3, 6)
                    u_list.append(u_batch.mean(dim=0))  # Baseline workflow step.
                u = torch.stack(u_list)  # (B, 6)
            elif u.size(0) == batch_size:
                # Baseline workflow step.
                if u.size(1) == 6:
                    # Baseline workflow step.
                    pass
                elif u.size(1) == 18:
                    # Baseline workflow step.
                    u = u.view(batch_size, 3, 6).mean(dim=1)  # (B, 6)
                elif u.size(1) % 6 == 0:
                    # Baseline workflow step.
                    num_mols = u.size(1) // 6
                    u = u.view(batch_size, num_mols, 6).mean(dim=1)  # (B, 6)
                else:
                    raise ValueError(f" unable to process global_features shape : {u.shape}, batch_size={batch_size}, expected number Two dimensions yes 6 multiple ")
            else:
                # Baseline workflow step.
                if u.size(0) % batch_size == 0:
                    mols_per_sample = u.size(0) // batch_size
                    if u.size(1) == 6:
                        u = u.view(batch_size, mols_per_sample, 6).mean(dim=1)  # (B, 6)
                    else:
                        raise ValueError(f" unable to process global_features shape : {u.shape}, batch_size={batch_size}")
                else:
                    raise ValueError(f"global_features number M dimensions ({u.size(0)}) unable to by batch_size ({batch_size}) Divide by ")
        else:
            # Baseline workflow step.
            if u.size(0) >= 3 and u.size(1) == 6:
                u = u[:3].mean(dim=0, keepdim=True)  # (1, 6)
            elif u.size(0) == 1 and u.size(1) == 18:
                u = u.view(1, 3, 6).mean(dim=1)  # (1, 6)
            else:
                u = u.mean(dim=0, keepdim=True) if u.size(0) > 0 else torch.zeros(1, 6, device=x.device)
        
        # Baseline workflow step.
        if u.dim() != 2 or u.size(1) != 6:
            raise ValueError(f" process after global_features shape Incorrect : {u.shape}, expected shape is (B, 6)")
        
        # Save the generated artifacts.
        u_global = u  # Baseline workflow step.
        
        # Baseline workflow step.
        u = self.global_proj(u)  # (B, hidden_dim)
        
        # Baseline workflow step.
        for layer in self.intra_layers:
            x, edge_attr = layer(x, edge_index, edge_attr, u_global, batch)
        
        # Baseline workflow step.
        # Baseline workflow step.
        if molecule_ranges is not None and len(molecule_ranges) > 0:
            # Baseline workflow step.
            if batch is not None:
                batch_size = batch.max().item() + 1
                x_updated_list = []
                for b in range(batch_size):
                    batch_mask = batch == b
                    x_b = x[batch_mask]
                    
                    # Baseline workflow step.
                    global_indices = torch.where(batch_mask)[0]
                    
                    # Baseline workflow step.
                    local_ranges = []
                    for start, end in molecule_ranges:
                        local_mask = (global_indices >= start) & (global_indices < end)
                        if local_mask.any():
                            local_indices = global_indices[local_mask]
                            local_start = local_indices.min().item()
                            local_end = local_indices.max().item() + 1
                            local_ranges.append((local_start, local_end))
                    
                    # Baseline workflow step.
                    if len(local_ranges) > 1:
                        inter_edges = []
                        for i, (start_i, end_i) in enumerate(local_ranges):
                            for j, (start_j, end_j) in enumerate(local_ranges):
                                if i != j:
                                    # Baseline workflow step.
                                    batch_edge_mask = batch_mask[edge_index[0]] & batch_mask[edge_index[1]]
                                    if batch_edge_mask.any():
                                        batch_edges = edge_index[:, batch_edge_mask]
                                        inter_mask = ((batch_edges[0] >= start_i) & (batch_edges[0] < end_i) &
                                                     (batch_edges[1] >= start_j) & (batch_edges[1] < end_j))
                                        if inter_mask.any():
                                            inter_edges.append(batch_edges[:, inter_mask])
                        
                        if inter_edges:
                            inter_edge_index = torch.cat(inter_edges, dim=1)
                            # Baseline workflow step.
                            local_inter_edges = []
                            for idx in range(inter_edge_index.size(1)):
                                src, dst = inter_edge_index[:, idx]
                                local_src = torch.where(global_indices == src)[0][0]
                                local_dst = torch.where(global_indices == dst)[0][0]
                                local_inter_edges.append([local_src.item(), local_dst.item()])
                            
                            if local_inter_edges:
                                local_inter_edge_index = torch.tensor(local_inter_edges, dtype=torch.long, device=x.device).T
                                x_b = self.inter_layer(x_b, local_inter_edge_index, local_ranges)
                    
                    x_updated_list.append(x_b)
                
                # Baseline workflow step.
                x = torch.cat(x_updated_list, dim=0)
            else:
                # Baseline workflow step.
                inter_edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
                for i, (start_i, end_i) in enumerate(molecule_ranges):
                    for j, (start_j, end_j) in enumerate(molecule_ranges):
                        if i != j:
                            mask = ((edge_index[0] >= start_i) & (edge_index[0] < end_i) &
                                    (edge_index[1] >= start_j) & (edge_index[1] < end_j))
                            inter_edge_mask |= mask
                
                if inter_edge_mask.any():
                    inter_edge_index = edge_index[:, inter_edge_mask]
                    x = self.inter_layer(x, inter_edge_index, molecule_ranges)
        
        # Baseline workflow step.
        x_explained, edge_attr_explained = self.explainer(x, edge_attr)
        
        # Baseline workflow step.
        for layer in self.post_explain_layers:
            x_explained, edge_attr_explained = layer(x_explained, edge_index, edge_attr_explained, u_global, batch)
        
        # Baseline workflow step.
        batch_size = batch.max().item() + 1 if batch is not None else 1
        num_molecules = 3  # Baseline workflow step.
        
        molecule_features = []
        for b in range(batch_size):
            if batch is not None:
                batch_mask = batch == b
                if not batch_mask.any():
                    molecule_features.append(torch.zeros(num_molecules, self.hidden_dim * 2, device=x.device))
                    continue
                x_batch = x_explained[batch_mask]
            else:
                x_batch = x_explained
            
            # Baseline workflow step.
            if molecule_ranges is not None and len(molecule_ranges) > 0:
                mol_feats = []
                node_idx = 0
                for start, end in molecule_ranges:
                    if batch is not None:
                        # Baseline workflow step.
                        batch_node_indices = torch.where(batch_mask)[0]
                        mol_node_mask = (batch_node_indices >= start) & (batch_node_indices < end)
                        if mol_node_mask.any():
                            mol_x = x_batch[mol_node_mask]
                        else:
                            mol_x = torch.zeros(0, self.hidden_dim, device=x.device)
                    else:
                        mol_x = x_batch[start:end]
                    
                    if mol_x.size(0) > 0:
                        # Baseline workflow step.
                        mol_batch = torch.zeros(mol_x.size(0), dtype=torch.long, device=x.device)
                        mol_feat = self.set2set(mol_x, mol_batch)
                        mol_feats.append(mol_feat.squeeze(0))
                    else:
                        mol_feats.append(torch.zeros(self.hidden_dim * 2, device=x.device))
                
                # Baseline workflow step.
                while len(mol_feats) < num_molecules:
                    mol_feats.append(torch.zeros(self.hidden_dim * 2, device=x.device))
                
                molecule_features.append(torch.stack(mol_feats[:num_molecules]))
            else:
                # Baseline workflow step.
                if x_batch.size(0) > 0:
                    avg_feat = x_batch.mean(dim=0)
                    # Baseline workflow step.
                    max_feat = x_batch.max(dim=0)[0]
                    combined_feat = torch.cat([avg_feat, max_feat], dim=0)
                    molecule_features.append(torch.stack([combined_feat] * num_molecules))
                else:
                    molecule_features.append(torch.zeros(num_molecules, self.hidden_dim * 2, device=x.device))
        
        # Baseline workflow step.
        molecule_features = torch.stack(molecule_features)  # (B, 3, hidden_dim*2)
        molecule_features = molecule_features.view(batch_size, -1)  # (B, 3*hidden_dim*2)
        
        # Configure experiment parameters.
        T_t_features = torch.stack([T_norm, t], dim=1)  # (B, 2)
        final_features = torch.cat([molecule_features, T_t_features], dim=1)
        
        # Generate model predictions.
        output = self.predictor(final_features)
        output = self._phase_softmax(output)
        
        if return_explanation:
            return output, x_explained, edge_attr_explained
        return output

