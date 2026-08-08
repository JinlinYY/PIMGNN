# -*- coding: utf-8 -*-
"""Implement the mmgnn dataset baseline module."""

from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from .graph_builder import MoleculeGraphBuilder
import os

from psmi_baselines.common.utils import Scaler


class MMGNNDataset(Dataset):
    """Represent the MMGNNDataset baseline component."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        T_scaler: Scaler,
        graph_builder: Optional[MoleculeGraphBuilder] = None,
        precompute: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.T_scaler = T_scaler
        self.graph_builder = graph_builder or MoleculeGraphBuilder()
        self.precompute = precompute
        
        self._graphs: Optional[list] = None
        self._T_norm: Optional[torch.Tensor] = None
        self._t: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None
        
        if self.precompute:
            self._build_cache()
    
    def _build_cache(self):
        """Run the build cache baseline operation."""
        graphs = []
        T_norm_list = []
        t_list = []
        y_list = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            
            # Baseline workflow step.
            graph1 = self.graph_builder.smiles_to_graph(row["smiles1"])
            graph2 = self.graph_builder.smiles_to_graph(row["smiles2"])
            graph3 = self.graph_builder.smiles_to_graph(row["smiles3"])
            
            # Baseline workflow step.
            merged_graph = self.graph_builder.merge_graphs([graph1, graph2, graph3])
            
            # Baseline workflow step.
            # Baseline workflow step.
            mol_ranges = merged_graph['molecule_ranges']
            data = Data(
                x=torch.from_numpy(merged_graph['node_features']).float(),
                edge_index=torch.from_numpy(merged_graph['edge_index']).long(),
                edge_attr=torch.from_numpy(merged_graph['edge_features']).float(),
                global_features=torch.from_numpy(merged_graph['global_features']).float(),
            )
            # Baseline workflow step.
            data.molecule_ranges = mol_ranges
            data.num_molecules = len(mol_ranges)
            graphs.append(data)
            
            # Baseline workflow step.
            T_norm = self.T_scaler.transform(
                pd.Series([row["T"]]).values.astype('float32')
            )[0]
            T_norm_list.append(float(T_norm))
            t_list.append(float(row["t"]))
            
            # Baseline workflow step.
            y = torch.tensor([
                row["Ex1"], row["Ex2"], row["Ex3"],
                row["Rx1"], row["Rx2"], row["Rx3"]
            ], dtype=torch.float32)
            y_list.append(y)
        
        self._graphs = graphs
        self._T_norm = torch.tensor(T_norm_list, dtype=torch.float32)
        self._t = torch.tensor(t_list, dtype=torch.float32)
        self._y = torch.stack(y_list)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self._graphs is not None:
            return (
                self._graphs[idx],
                self._T_norm[idx],
                self._t[idx],
                self._y[idx]
            )
        
        # Baseline workflow step.
        row = self.df.iloc[idx]
        
        graph1 = self.graph_builder.smiles_to_graph(row["smiles1"])
        graph2 = self.graph_builder.smiles_to_graph(row["smiles2"])
        graph3 = self.graph_builder.smiles_to_graph(row["smiles3"])
        
        merged_graph = self.graph_builder.merge_graphs([graph1, graph2, graph3])
        
        data = Data(
            x=torch.from_numpy(merged_graph['node_features']).float(),
            edge_index=torch.from_numpy(merged_graph['edge_index']).long(),
            edge_attr=torch.from_numpy(merged_graph['edge_features']).float(),
            global_features=torch.from_numpy(merged_graph['global_features']).float(),
        )
        data.molecule_ranges = merged_graph['molecule_ranges']
        data.num_molecules = len(merged_graph['molecule_ranges'])
        
        T_norm = self.T_scaler.transform(
            pd.Series([row["T"]]).values.astype('float32')
        )[0]
        
        y = torch.tensor([
            row["Ex1"], row["Ex2"], row["Ex3"],
            row["Rx1"], row["Rx2"], row["Rx3"]
        ], dtype=torch.float32)
        
        return data, torch.tensor(T_norm, dtype=torch.float32), torch.tensor(row["t"], dtype=torch.float32), y


def collate_fn(batch):
    """Run the collate fn baseline operation."""
    graphs, T_norm, t, y = zip(*batch)
    
    # Baseline workflow step.
    batch_graphs = Batch.from_data_list(graphs)
    
    T_norm = torch.stack(T_norm)
    t = torch.stack(t)
    y = torch.stack(y)
    
    return batch_graphs, T_norm, t, y

