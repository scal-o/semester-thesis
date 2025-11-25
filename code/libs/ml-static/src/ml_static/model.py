from pathlib import Path
from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from ml_static.encoders import REncoder, VEncoder


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_features):
        origin, destination = edge_index

        # concatenate: origin + destination features + connecting edge features
        z = torch.cat([x[origin], x[destination], edge_features], dim=1)
        z = self.lin1(z)
        z = F.leaky_relu(z)
        z = self.lin2(z)
        z = F.leaky_relu(z)
        return z.view(-1)


class GNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()

        # Store initialization parameters for serialization
        self._init_params = {  # type: ignore
            "input_dim": input_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
        }

        self.vencoder1 = VEncoder(input_channels, hidden_channels, 2)
        self.vencoder2 = VEncoder(hidden_channels, hidden_channels, 2)

        self.rencoder1 = REncoder(hidden_channels, hidden_channels, 2)
        self.rencoder2 = REncoder(hidden_channels, hidden_channels, 2)

        self.mlp = MLPRegressor(hidden_channels * 2 + 2, out_channels)

    def forward(self, graph: HeteroData):
        # apply transformer layers to get node embeddings
        g = self.vencoder1(graph.x_dict["nodes"], graph["virtual"].edge_index)
        g = self.vencoder2(g, graph["virtual"].edge_index)
        g = self.rencoder1(g, graph["real"].edge_index, graph["real"].edge_features)
        g = self.rencoder2(g, graph["real"].edge_index, graph["real"].edge_features)

        # create link predictions with the MLP
        z = self.mlp(g, graph["real"].edge_index, graph["real"].edge_features)

        return z

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path | str) -> Self:
        """
        Load a GNN model from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.pt).
            device: Device to load the model on ("cpu" or "cuda").

        Returns:
            Loaded GNN model in evaluation mode.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # extract initialization parameters
        init_params = checkpoint["init_params"]

        # instantiate the model
        model = cls(
            input_channels=init_params["input_dim"],
            hidden_channels=init_params["hidden_channels"],
            out_channels=init_params["out_channels"],
        )

        # load the state dict
        model.load_state_dict(checkpoint["state_dict"])

        return model
