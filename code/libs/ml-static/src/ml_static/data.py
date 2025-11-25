from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import List

import click
import geopandas as gpd
import numpy as np
import openmatrix as omx
import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm


class STADataset(Dataset):
    """
    Pytorch Geometric dataset for static traffic assignment data.

    Each sample in the dataset is a graph representing the network state
    from a single scenario's static traffic assignment results.

    The node and link features closely resemble the ones presented in Liu & Meidani (2024).
    """

    def __init__(self, network_dir: Path | str, **kwargs):
        """
        Args:
            root (str): Root directory where the dataset should be saved.
                        This directory is split into 'raw' and 'processed' subdirectories.
            transform (callable, optional): A function/transform that takes in an
                                           `torch_geometric.data.Data` object and returns a
                                           transformed version. The data object will be
                                           transformed before every access. (default: None)
            pre_transform (callable, optional): A function/transform that takes in
                                                an `torch_geometric.data.Data` object and returns
                                                a transformed version. The data object will be
                                                transformed before being saved to disk.
                                                (default: None)
        """
        # create root directory for the dataset inside the network directory
        # i.e. alongside the other scenarios_* directories
        self.root = Path(network_dir) / "scenarios_sta_mldata"
        if not self.root.is_dir():
            self.root.mkdir()

        # define dirs
        self.network_dir = Path(network_dir)
        self.node_data_dir = self.network_dir / "scenarios_geojson"
        self.link_data_dir = self.network_dir / "scenarios_sta_results"

        # define scenario names
        # retrieve all available scenarios and results
        scenarios = list(self.node_data_dir.glob("scenario_*"))
        results = list(self.link_data_dir.glob("scenario_*"))

        # remove scenario 0 from both scenario and results
        if (self.node_data_dir / "scenario_00000").is_dir():
            scenarios.remove(self.node_data_dir / "scenario_00000")
        if (self.link_data_dir / "scenario_00000").is_dir():
            results.remove(self.link_data_dir / "scenario_00000")

        # sort paths deterministically by name to ensure reproducible ordering
        # across different systems and filesystem implementations
        scenarios = sorted(scenarios, key=lambda p: p.name)
        results = sorted(results, key=lambda p: p.name)

        # check there are the same number of scenarios and results
        if len(scenarios) != len(results):
            raise ValueError(
                f"Found {len(scenarios)} scenario files but {len(results)} results files."
            )

        self.scenario_paths = scenarios
        self.result_paths = results
        self.scenario_names = [scenario.name for scenario in scenarios]

        # init base class
        super().__init__(str(self.root), **kwargs)

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the list of files that must be present in the raw_dir directory
        in order to skip the download step.
        The file names are derived from the scenarios available in the
        scenarios_sta_results directory.
        """
        scenario_names = self.scenario_paths

        # to create a complete graph we need:
        # - node coordinates (from nodes.geojson)
        # - od data (from omx matrix or geojson)
        names_nodes = [f"{scenario.name}_nodes.geojson" for scenario in scenario_names]
        names_od = [f"{scenario.name}_od.omx" for scenario in scenario_names]
        names_links = [f"{scenario.name}.parquet" for scenario in scenario_names]

        return names_nodes + names_od + names_links

    @property
    def processed_file_names(self) -> List[str]:
        """
        Return the list of files that must be present in the processed_dir directory
        in order to skip the processing step.
        The file names are derived from the scenarios available in the
        scenarios_sta_results directory.
        """
        scenario_names = self.scenario_paths

        # we expect to find a single .pt file for every scenario
        names = [f"{scenario.name}.pt" for scenario in scenario_names]

        return names

    def download(self):
        """
        The "download" in this case is limited to making a copy of node files (node
        coordinates and od) and result files to the raw directory with a new name structure.
        """

        print("--- Dataset Download ---")
        print(f"Network: {self.network_dir.name}")
        print(f"Path: {self.network_dir}")

        scenario_paths = self.scenario_paths
        result_paths = self.result_paths

        # raw dir
        dest = Path(self.raw_dir)

        for scenario in tqdm(scenario_paths, desc="Copying node data"):
            shutil.copy(
                scenario / "nodes.geojson",
                dest / f"{scenario.name}_nodes.geojson",
            )
            shutil.copy(scenario / "od.omx", dest / f"{scenario.name}_od.omx")

        for scenario in tqdm(result_paths, desc="Copying result data"):
            shutil.copy(
                scenario / f"{scenario.name}.parquet",
                dest / f"{scenario.name}.parquet",
            )

        print("--- Dataset Download Complete ---")

    def process(self):
        """
        Processes raw data and saves it into the `processed_dir`.
        """

        print("--- Dataset Processing ---")
        print(f"Network: {self.network_dir.name}")
        print(f"Path: {self.network_dir}")

        scenario_names = self.scenario_names

        for scenario in tqdm(scenario_names, desc="Processing data"):
            # raw dir
            src = Path(self.raw_dir)

            ## 1. Nodes
            # read node coordinates
            node_df = gpd.read_file(src / f"{scenario}_nodes.geojson")
            node_coords = np.array(node_df[["x", "y"]])

            # read od matrix
            omx_mat = omx.open_file(src / f"{scenario}_od.omx")
            od_mat = np.array(omx_mat["matrix"])

            # get full od matrix (od_mat only contains centrid data)
            full_mat = pd.DataFrame(od_mat)
            full_mat = full_mat.reindex(
                index=range(0, len(node_df)),
                columns=range(0, len(node_df)),
                fill_value=0.0,
            )

            # scale demand matrix between 0 and 100
            full_mat = (full_mat - full_mat.min()) / (full_mat.max() - full_mat.min()) * 100

            full_mat = full_mat.values

            # concatenate node coordinates and od matrix to create node features
            node_features = np.concat((full_mat, node_coords), axis=1)

            # convert to tensors
            node_coords = torch.tensor(node_coords, dtype=torch.float)
            node_features = torch.tensor(node_features, dtype=torch.float)

            ##
            ## 2. Real Links
            # read links from parquet file
            # we can read them using pandas (instead of geopandas) as we don't
            # need the geometry data
            link_df = pd.read_parquet(src / f"{scenario}.parquet")

            # create edge index (0-based index for pytorch)
            edges = link_df[["a_node", "b_node"]].values
            edges -= 1

            # extract edge features and labels
            edge_features = link_df[["capacity", "free_flow_time"]].values
            edge_vcr = link_df["volume_capacity_ratio"].values
            edge_flow = link_df["flow"].values

            # standardize edge features (z-score normalization)
            edge_features = (edge_features - edge_features.mean(axis=0)) / (
                edge_features.std(axis=0)
            )

            # convert to tensors
            edges = torch.tensor(edges, dtype=torch.long).t()
            edge_features = torch.tensor(edge_features, dtype=torch.float)
            edge_vcr = torch.tensor(edge_vcr, dtype=torch.float)
            edge_flow = torch.tensor(edge_flow, dtype=torch.float)

            ##
            ## 3. Virtual Links
            # we gather the virtual links from the od matrix
            dmat = pd.DataFrame(od_mat)

            # add origin col
            dmat.insert(0, "origin", np.array(range(0, len(dmat))))

            # melt dataframe
            dmat = dmat.melt(
                id_vars="origin",
                value_vars=dmat.columns[1:],
                var_name="destination",
            )
            dmat = dmat.loc[dmat["value"] > 0]
            dmat = dmat[["origin", "destination"]]
            dmat = dmat.astype("int32")

            # create edge index
            virtual_edges = dmat.values

            # convert to tensor
            virtual_edges = torch.tensor(virtual_edges, dtype=torch.long).t()

            # close matrix
            omx_mat.close()

            ##
            ## 4. Data
            data = HeteroData()

            # append node data
            data["nodes"].x = node_features
            data["nodes"].pos = node_coords

            # append real edge data
            data["nodes", "real", "nodes"].edge_index = edges
            data["nodes", "real", "nodes"].edge_features = edge_features
            data["nodes", "real", "nodes"].edge_vcr = edge_vcr
            data["nodes", "real", "nodes"].edge_flow = edge_flow

            # append virtual edge data
            data["nodes", "virtual", "nodes"].edge_index = virtual_edges

            # save data to disk
            try:
                data.validate()
            except AssertionError as e:
                raise AssertionError(f"Data validation failed: {e}") from e

            torch.save(data, Path(self.processed_dir) / f"{scenario}.pt")

        print("--- Dataset Processing Complete ---")

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:
        scenario = self.scenario_names[idx]
        data = torch.load(Path(self.processed_dir) / f"{scenario}.pt", weights_only=False)
        return data

    def __getitem__(self, idx):
        """
        Handles indexing and slicing of the dataset.
        If an integer is passed, it returns a single data object.
        If a slice or list of indices is passed, it returns a `Subset` object
        with the sliced custom attributes attached.
        """
        if isinstance(idx, int):
            data = self.get(idx)
            data = data if self.transform is None else self.transform(data)
            return data

        # return a subset of the original dataset if a list or slice is passed
        subset = copy.copy(self)

        # monkey-patch the subset to include the correct indices and custom attributes
        # so it behaves like a normal STADataset for our purposes
        indices = subset.indices()
        subset._indices = [indices[i] for i in idx]

        subset.scenario_names = [self.scenario_names[i] for i in subset.indices()]
        subset.scenario_paths = [self.scenario_paths[i] for i in subset.indices()]
        subset.result_paths = [self.result_paths[i] for i in subset.indices()]

        return subset


def create_splits(
    dataset: STADataset, split: tuple[float, float, float]
) -> tuple[
    tuple[STADataset, STADataset, STADataset],
    tuple[list, list, list],
]:
    """
    Utility function to split dataset into train, val and test sets.

    Args:
        dataset: The original dataset to split.
        split: Tuple containing the relative sizes of the train, val and test sets.

    Results:
        The three resulting STAdatasets (train, val, test), and tuples containing
        the indices used for the split.
    """

    # normalize the sum of the splits
    split_total = sum(split)
    split = (
        split[0] / split_total,
        split[1] / split_total,
        split[2] / split_total,
    )

    # determine sizes
    n_total = len(dataset)
    n_train = int(n_total * split[0])
    n_val = int(n_total * split[1])

    # generate indices
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train : n_train + n_val].tolist()
    test_indices = indices[n_train + n_val :].tolist()

    # create subsets
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    ), (
        train_indices,
        val_indices,
        test_indices,
    )


class VarTransform(BaseTransform):
    """
    Custom, variable transformation for Pytorch Geometric datasets.
    """

    def __init__(self, target: tuple, transform: str | None = None):
        """
        Args:
            type: List specifying the target type (e.g. real links = ["nodes", "real", "nodes"]).
            label: The target label to extract.
            transform: The transform to apply to each data object. If none, the data defined by
                "label" is set as the target without further transformations.
        """
        self.type, self.label = target
        self.transform = transform

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Applies the transformation to the data object.

        Args:
            data: The input HeteroData object.

        Returns:
            The transformed HeteroData object with the target extracted.
        """
        target = data[self.type].get(self.label, None)
        if target is None:
            raise KeyError(f"Target label '{self.label}' not found in data type '{self.type}'.")

        if self.transform is not None:
            if self.transform == "log":
                target = torch.log1p(target)
            else:
                raise NotImplementedError(f"Unknown transform '{self.transform}' specified.")

        data.y = target
        return data


@click.command()
@click.argument("network")
@click.option(
    "--path",
    default="data",
    show_default=True,
    help="The base path to the networks directory.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-creation of the dataset.",
)
def create_dataset(
    network: str,
    path: Path | str | None = None,
    force: bool = False,
):
    """
    Creates a Pytorch Geometric dataset from NETWORK's static traffic assignment data.
    """
    print("--- Dataset Creation ---")
    print(f"Network: {network}")
    print(f"Base path: {path}")

    # setting up paths
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    network_path = path / network

    if not network_path.is_dir():
        raise ValueError(
            f"Network path {path} does not exist."
            "Make sure to run the script from the network directory or provide a base_path."
        )

    try:
        # initialize the dataset
        STADataset(network_path, force_reload=force)

        print("--- Dataset Creation Complete ---")

    except Exception as e:
        raise Exception(f"Dataset generation failed. An unexpected error occurred: {e}") from e
