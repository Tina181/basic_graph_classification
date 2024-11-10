import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(nn.Module):
    def __init__(self, dataset, hidden_channels=64):
        super(GCN, self).__init__()
        '''
        only need to know the number of features and the number of classes
        num_features: dimension of input node features -> D
        num_classes: number of output classes -> C
        '''
        self.conv1 = GCNConv(dataset.num_features, hidden_channels) # D -> hidden_channels
        self.conv2 = GCNConv(hidden_channels, hidden_channels)  # hidden_channels -> hidden_channels
        self.conv3 = GCNConv(hidden_channels, hidden_channels)  # hidden_channels -> hidden_channels
        self.linear = Linear(hidden_channels, dataset.num_classes)  # hidden_channels -> num_classes
        
        
    def forward(self, x, edge_index, batch):
        '''
        x: node features (N, D)
        edge_index: edge index (2, E)
        batch: batch index (N) -> show which nodes belong to which graph eg: (0, 0, 0, 1, 1, 1, 2, 2, 2) means 3 graphs with 3 nodes each
        '''
        # 1. conv layers
        x = self.conv1(x, edge_index) # (N, D) -> (N, hidden_channels)
        x = self.conv2(x, edge_index) # (N, hidden_channels) -> (N, hidden_channels)
        x = self.conv3(x, edge_index) # (N, hidden_channels) -> (N, hidden_channels)
        
        # 2. readout layer
        '''get the graph-level representation'''
        x = global_mean_pool(x, batch) # (num_graphs, hidden_channels): compute mean of node features in each graph
        
        # 3 classify layer
        x = F.dropout(x, p=0.5, training=self.training) # add dropout
        x = self.linear(x) # (num_graphs, hidden_channels) -> (num_graphs, num_classes)
        return x