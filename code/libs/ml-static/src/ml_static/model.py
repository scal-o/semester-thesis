from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_static.encoders import REncoder, VEncoder

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import Config


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

    def extract_checkpoint(self):
        """
        Extract model checkpoint containing current state dict.
        The checkpoint can be extended to include additional information as needed.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint = {
            "state_dict": self.state_dict(),
        }
        return checkpoint

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """
        Create GNN model from configuration object.

        Args:
            config: Configuration object.

        Returns:
            GNN model instance.
        """
        return cls(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.output_channels,
        )

    @classmethod
    def from_checkpoint(cls, config: Config, checkpoint_path: Path | str) -> Self:
        """
        Load a GNN model from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.pt).
            device: Device to load the model on ("cpu" or "cuda").

        Returns:
            Loaded GNN model in evaluation mode.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # instantiate the model
        model = cls.from_config(config)

        # load the state dict
        model.load_state_dict(checkpoint["state_dict"])

        return model
