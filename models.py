from typing import List

import torch
from torch import nn
from torch_geometric.nn import (
    GCNConv,            # Graph convolution layer
    BatchNorm,          # BatchNorm that understands graph batches
    global_mean_pool    # Readout: averages node embeddings per graph
)


class GCNGraphClassifier(nn.Module):
    """
    A simple, strong baseline for graph-level classification.

    Args:
        in_channels:   Node feature dimension (dataset.num_features)
        hidden_channels: Hidden size for all GCN layers and MLP
        num_layers:    Number of GCN blocks (>= 2 recommended)
        num_classes:   Number of graph classes (6 for ENZYMES)
        dropout:       Dropout probability used inside blocks and MLP
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        assert num_layers >= 2, "Use at least 2 GCN layers."

        # ----- Message passing trunk -----
        self.convs: List[GCNConv] = nn.ModuleList()
        self.bns:   List[BatchNorm] = nn.ModuleList()

        # First layer: input -> hidden
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers: hidden -> hidden
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # Activation + regularization used after each conv
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # ----- Graph-level readout + classifier -----
        # Readout: global_mean_pool turns variable-size node sets
        # into a fixed-size graph embedding per graph in the batch.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )

        # Initialize linear layers for stable training
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        Expects a PyG `Data` batch with:
          data.x          [N_total_nodes, in_channels]
          data.edge_index [2, E]
          data.batch      [N_total_nodes] (node->graph id)
        Returns:
          logits          [batch_size_in_graphs, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Stacked GCN blocks
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)   # message passing + linear transform
            x = bn(x)                 # stabilize across mini-batches of graphs
            x = self.act(x)           # nonlinearity
            x = self.drop(x)          # regularization

        # Graph-level readout
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]

        # Class logits
        out = self.mlp(x)               # [num_graphs, num_classes]
        return out