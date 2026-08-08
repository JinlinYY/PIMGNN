"""Implement the cgib models cgib baseline module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .gnn import MPNN, GIN
from torch_scatter import scatter

# ===== Set2Set (merged from set2set.py) =====
class Set2Set(nn.Module):
    """Represent the Set2Set baseline component."""
    def __init__(self, input_dim, processing_steps=3):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.processing_steps = processing_steps
        
        # Baseline workflow step.
        # Baseline workflow step.
        # Configure the output artifacts.
        self.lstm = nn.LSTM(input_dim * 2, input_dim, batch_first=False)
        
    def forward(self, x, batch):
        """Run the forward baseline operation."""
        batch_size = batch.max().item() + 1
        
        # Baseline workflow step.
        h = (torch.zeros(1, batch_size, self.input_dim, device=x.device, dtype=x.dtype),
             torch.zeros(1, batch_size, self.input_dim, device=x.device, dtype=x.dtype))
        
        q_star = None
        
        for step in range(self.processing_steps):
            # Baseline workflow step.
            if q_star is None:
                # Baseline workflow step.
                lstm_input = torch.zeros(1, batch_size, self.input_dim * 2, device=x.device, dtype=x.dtype)
            else:
                lstm_input = q_star.unsqueeze(0)  # [1, batch_size, input_dim * 2]
            
            # Baseline workflow step.
            q, h = self.lstm(lstm_input, h)
            q = q.squeeze(0)  # [batch_size, input_dim]
            
            # Baseline workflow step.
            q_expanded = q[batch]  # [num_nodes, input_dim]
            
            # Baseline workflow step.
            e = (x * q_expanded).sum(dim=1, keepdim=True)  # [num_nodes, 1]
            
            # Baseline workflow step.
            # Baseline workflow step.
            e_max = scatter(e, batch, dim=0, dim_size=batch_size, reduce='max')  # [batch_size, 1]
            e_exp = torch.exp(e - e_max[batch])  # [num_nodes, 1]
            e_sum = scatter(e_exp, batch, dim=0, dim_size=batch_size, reduce='sum')  # [batch_size, 1]
            alpha = e_exp / (e_sum[batch] + 1e-10)  # [num_nodes, 1]
            
            # Baseline workflow step.
            r = scatter(alpha * x, batch, dim=0, dim_size=batch_size, reduce='sum')  # [batch_size, input_dim]
            
            # Baseline workflow step.
            q_star = torch.cat([q, r], dim=1)  # [batch_size, input_dim * 2]
        
        return q_star




class CGIB(nn.Module):
    """Represent the CGIB baseline component."""
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers=3,
                 beta=1e-3,
                 temperature=1.0,
                 gnn_type='mpnn',
                 use_contrastive=False,
                 set2set_steps=3,
                 edge_dim=3,
                 constrain_output=False):
        """Run the init baseline operation."""
        super(CGIB, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.beta = beta
        self.temperature = temperature
        self.use_contrastive = use_contrastive
        self.constrain_output = constrain_output
        
        # Baseline workflow step.
        if gnn_type == 'mpnn':
            self.gnn1 = MPNN(input_dim, hidden_dim, hidden_dim, num_layers, edge_dim=edge_dim)
            self.gnn2 = MPNN(input_dim, hidden_dim, hidden_dim, num_layers, edge_dim=edge_dim)
        elif gnn_type == 'gin':
            self.gnn1 = GIN(input_dim, hidden_dim, hidden_dim, num_layers)
            self.gnn2 = GIN(input_dim, hidden_dim, hidden_dim, num_layers)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Set2Set readout
        self.readout1 = Set2Set(hidden_dim * 2, processing_steps=set2set_steps)
        self.readout2 = Set2Set(hidden_dim * 2, processing_steps=set2set_steps)
        
        # Baseline workflow step.
        self.importance_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Generate model predictions.
        # Configure the output artifacts.
        # z_combined = [z_G1_CIB, z_G2] = [hidden_dim * 4, hidden_dim * 4] = hidden_dim * 8
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # Baseline workflow step.
        # Configure the output artifacts.
        if not use_contrastive:
            self.variational_mlp = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        
    def compute_interaction(self, E1, E2):
        """Run the compute interaction baseline operation."""
        # Baseline workflow step.
        E1_norm = F.normalize(E1, p=2, dim=1)
        E2_norm = F.normalize(E2, p=2, dim=1)
        I = torch.mm(E1_norm, E2_norm.t())  # [N1, N2]
        
        # Baseline workflow step.
        E1_tilde = torch.mm(I, E2)  # [N1, d]
        E2_tilde = torch.mm(I.t(), E1)  # [N2, d]
        
        return I, E1_tilde, E2_tilde
    
    def sample_cib_graph(self, H1, batch1):
        """Run the sample cib graph baseline operation."""
        # Baseline workflow step.
        p_logits = self.importance_mlp(H1).squeeze(-1)  # [N1]
        p_values = torch.sigmoid(p_logits)  # [N1]
        
        # Baseline workflow step.
        u = torch.rand_like(p_values)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        lambda_logits = (torch.log(p_values + 1e-10) - torch.log(1 - p_values + 1e-10) + 
                         gumbel_noise) / self.temperature
        lambda_values = torch.sigmoid(lambda_logits)  # [N1]
        
        # Baseline workflow step.
        mu_H1 = H1.mean(dim=0, keepdim=True)  # [1, 2d]
        sigma_H1 = H1.std(dim=0, keepdim=True) + 1e-6  # [1, 2d]
        epsilon = torch.randn_like(H1) * sigma_H1 + mu_H1
        
        # Baseline workflow step.
        lambda_expanded = lambda_values.unsqueeze(1)  # [N1, 1]
        T1 = lambda_expanded * H1 + (1 - lambda_expanded) * epsilon
        
        return T1, lambda_values, p_values
    
    def compute_mi1_loss(self, lambda_values, H1):
        """Run the compute mi1 loss baseline operation."""
        N1 = lambda_values.size(0)
        A = ((1 - lambda_values) ** 2).sum()
        
        mu_H1 = H1.mean(dim=0)
        sigma_H1 = H1.std(dim=0) + 1e-6
        
        B = (lambda_values.unsqueeze(1) * (H1 - mu_H1.unsqueeze(0)) / (sigma_H1.unsqueeze(0) + 1e-6)).sum()
        
        loss = -0.5 * torch.log(A + 1e-10) + 0.5 / N1 * A + 0.5 / N1 * (B ** 2)
        
        return loss
    
    def compute_mi2_loss(self, z_G1_CIB, z_G2, batch_size):
        """Run the compute mi2 loss baseline operation."""
        if self.use_contrastive:
            # Baseline workflow step.
            # Baseline workflow step.
            sim_matrix = torch.mm(F.normalize(z_G1_CIB, p=2, dim=1),
                                 F.normalize(z_G2, p=2, dim=1).t()) / self.temperature
            
            # Baseline workflow step.
            labels = torch.arange(batch_size, device=z_G1_CIB.device)
            loss = F.cross_entropy(sim_matrix, labels)
            
            return loss
        else:
            # Baseline workflow step.
            # Generate model predictions.
            z_G2_pred = self.variational_mlp(z_G1_CIB)
            
            # Baseline workflow step.
            loss = F.mse_loss(z_G2_pred, z_G2)
            
            return loss
    
    def forward(self, data1, data2, return_loss_components=False):
        """Run the forward baseline operation."""
        # Baseline workflow step.
        E1 = self.gnn1(data1.x, data1.edge_index, data1.edge_attr if hasattr(data1, 'edge_attr') else None)  # [N1, d]
        E2 = self.gnn2(data2.x, data2.edge_index, data2.edge_attr if hasattr(data2, 'edge_attr') else None)  # [N2, d]
        
        # Baseline workflow step.
        I, E1_tilde, E2_tilde = self.compute_interaction(E1, E2)
        
        # Baseline workflow step.
        H1 = torch.cat([E1, E1_tilde], dim=1)  # [N1, 2d]
        H2 = torch.cat([E2, E2_tilde], dim=1)  # [N2, 2d]
        
        # Baseline workflow step.
        T1, lambda_values, p_values = self.sample_cib_graph(H1, data1.batch)
        
        # Baseline workflow step.
        z_G1 = self.readout1(H1, data1.batch)  # [batch_size, 2d]
        z_G2 = self.readout2(H2, data2.batch)  # [batch_size, 2d]
        z_G1_CIB = self.readout1(T1, data1.batch)  # [batch_size, 2d]
        
        # Generate model predictions.
        z_combined = torch.cat([z_G1_CIB, z_G2], dim=1)  # [batch_size, hidden_dim * 8]
        pred = self.predictor(z_combined)  # [batch_size, output_dim]
        
        # Configure the output artifacts.
        if self.constrain_output:
            pred = torch.sigmoid(pred)  # Configure the output artifacts.
        
        if return_loss_components:
            # Compute the training loss.
            batch_size = z_G1.size(0)
            mi1_loss = self.compute_mi1_loss(lambda_values, H1)
            mi2_loss = self.compute_mi2_loss(z_G1_CIB, z_G2, batch_size)
            
            loss_components = {
                'mi1': mi1_loss,
                'mi2': mi2_loss,
                'lambda': lambda_values,
                'p': p_values
            }
            
            return pred, loss_components
        
        return pred
    
    def compute_loss(self, pred, target, loss_components, use_supervised=True):
        """Run the compute loss baseline operation."""
        # Generate model predictions.
        if self.output_dim == 1:
            # Baseline workflow step.
            pred_loss = F.mse_loss(pred.squeeze(), target)
        else:
            # Baseline workflow step.
            pred_loss = F.cross_entropy(pred, target.long())
        
        # Compute the training loss.
        sup_loss = 0.0
        if use_supervised:
            # Baseline workflow step.
            # Generate model predictions.
            # Baseline workflow step.
            pass
        
        # Compute the training loss.
        total_loss = pred_loss + self.beta * (loss_components['mi1'] + loss_components['mi2'])
        
        if use_supervised:
            total_loss = total_loss + sup_loss
        
        return total_loss, {
            'pred_loss': pred_loss,
            'mi1_loss': loss_components['mi1'],
            'mi2_loss': loss_components['mi2'],
            'total_loss': total_loss
        }

