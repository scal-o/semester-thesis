import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.data import HeteroData


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
        z = F.relu(z)
        z = self.lin2(z)
        return z.view(-1)


class Conv(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = pygnn.GCNConv(input_channels, hidden_channels)
        self.conv2 = pygnn.GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GNN(nn.Module):
    def __init__(self, data_sample, hidden_channels, out_channels):
        super().__init__()

        self.graph_conv = Conv(data_sample.num_features["nodes"], hidden_channels, out_channels)
        self.graph_conv = pygnn.to_hetero(self.graph_conv, data_sample.metadata(), aggr="sum")
        self.mlp = MLPRegressor(out_channels * 2 + 2, out_channels)

    def forward(self, graph: HeteroData):
        # pass graph features as dict to the HeteroGNN
        g = self.graph_conv(graph.x_dict, graph.edge_index_dict)

        # create link predictions with the MLP
        z = self.mlp(g["nodes"], graph["real"].edge_index, graph["real"].edge_features)

        return z


def train(model, optimizer, criterion, graph):
    model.train()
    optimizer.zero_grad()

    pred = model(graph)
    loss = criterion(pred, graph["real"].edge_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(model, criterion, graph):
    model.eval()
    with torch.no_grad():
        pred = model(graph)
        loss = criterion(pred, graph["real"].edge_labels)

    return loss.item()
