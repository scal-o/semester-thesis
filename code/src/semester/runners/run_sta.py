from pathlib import Path
from contextlib import redirect_stderr
import io


from functools import partial
import click
import geopandas as gpd
import pandas as pd
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.traffic_assignment import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
from tqdm import tqdm
import logging

logger = logging.getLogger("aequilibrae")
logger.setLevel(logging.ERROR)

CENTROID_FLOW_BLOCKING = {
    "anaheim": True,
    "sioux_falls": False,
}


def find_centroids(scenario_gpkg_path: Path):
    """
    Small function to find centroids for a given scenario (from OD matrix).
    """
    od_df = gpd.read_file(scenario_gpkg_path, layer="od")

    centroids = pd.concat([od_df["origin"], od_df["destination"]])
    centroids = centroids.unique()
    return centroids


def load_network(scenario_gpkg_path: Path, block_centroid_flows: False):
    """
    Loader for network graphs.

    Loads graphs from scenario gpkg files.
    """

    # load scenario data
    if not scenario_gpkg_path.exists():
        raise FileNotFoundError(f"Scenario '{scenario_gpkg_path}' not found. ")

    # load network centroids
    centroids = find_centroids(scenario_gpkg_path)

    # load network data
    net_df = gpd.read_file(scenario_gpkg_path, layer="links")
    net_df = net_df[
        [
            "link_id",
            "init_node",
            "term_node",
            "capacity",
            "free_flow_time",
            "b",
            "power",
            "geometry",
        ]
    ]
    net_df = net_df.rename(columns={"init_node": "a_node", "term_node": "b_node"})
    net_df = net_df.assign(direction=1)

    # load node geometry data
    nodes = gpd.read_file(scenario_gpkg_path, layer="nodes")
    nodes = nodes[["node_id", "x", "y"]]
    nodes.index = nodes["node_id"].copy()
    nodes = nodes.rename(columns={"x": "lat", "y": "lon"})

    # create graph from network dataframe
    graph = Graph()
    graph.network = net_df
    graph.prepare_graph(centroids)

    graph.set_graph("free_flow_time")
    graph.capacity = net_df["capacity"].values
    graph.free_flow_time = net_df["free_flow_time"].values

    # graph.set_skimming(["free_flow_time"])
    graph.set_blocked_centroid_flows(block_centroid_flows)
    graph.network["id"] = graph.network["link_id"]
    graph.lonlat_index = nodes.loc[graph.all_nodes]

    return graph


def load_matrix(scenario_omx_path: Path):
    """
    Loader for OD matrices.

    Loads matrices in AequilibraE matrix objects from OMX files.
    """

    # load scenario data
    if not scenario_omx_path.exists():
        raise FileNotFoundError(f"Matrix '{scenario_omx_path}' not found. ")

    # create AequilibraE matrix from OMX
    mat = AequilibraeMatrix()
    mat.create_from_omx(str(scenario_omx_path))
    mat.computational_view(["matrix"])

    return mat


def run_assignment(
    scenario_name: str, scenarios_dir: Path, output_dir=Path, block_centroid_flows=False
):
    """
    Loader for a complete Assignment problem.

    Loads network and od matrix from a scenario and creates a TrafficAssignment instance.
    """

    # load graph and od matrix
    scenario_gpkg_path = scenarios_dir / "scenarios_gpkg" / f"{scenario_name}.gpkg"
    scenario_omx_path = scenarios_dir / "scenarios_omx" / f"{scenario_name}.omx"

    graph = load_network(scenario_gpkg_path, block_centroid_flows)
    matrix = load_matrix(scenario_omx_path)

    # create traffic class
    traffic_class = TrafficClass("c", graph, matrix)

    # create traffic assignment object
    assignment = TrafficAssignment()
    assignment.set_classes([traffic_class])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.set_algorithm("bfw")
    assignment.max_iter = 500
    assignment.rgap_target = 1e-5
    assignment.set_cores(1)

    # run traffic assignment
    with io.StringIO() as s:
        # with open("log.txt", "w") as s:
        with redirect_stderr(s):
            assignment.execute()

    # assignment.execute()

    # retrieve assignment results
    results = assignment.results()
    network = graph.network.copy()
    network.index = network["id"].values

    # add resulting flows to network dataset
    network["flow"] = results["PCE_AB"]
    network["volume_capacity_ratio"] = results["VOC_AB"]
    network["congested_time"] = results["Congested_Time_AB"]

    # save to file
    scenario_results_path = output_dir / f"{scenario_name}.parquet"
    scenario_convergence_path = output_dir / "report" / f"{scenario_name}_convergence.parquet"

    network.to_parquet(scenario_results_path)
    assignment.report().to_parquet(scenario_convergence_path)

    return True


@click.command()
@click.argument("network")
@click.option(
    "--path", default="data", show_default=True, help="The base path to the scenarios directory."
)
@click.option(
    "--results-dir",
    default="scenarios_sta_results",
    show_default=True,
    help="The name to give to the results directory.",
)
def run_sta(
    network: str,
    path: str,
    results_dir: str,
):
    """
    Runs static assignment using AequilibraE.
    """
    print("--- Running static traffic assignment ---")
    print(f"Network: {network}")
    print(f"Path: {path}")

    # setting up paths
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    if results_dir is None:
        results_dir = "scenarios_sta_results"

    scenarios_path = path / network
    output_path = scenarios_path / results_dir
    output_reports_path = output_path / "report"

    # check paths

    if not scenarios_path.is_dir():
        raise ValueError(
            f"Scenarios path {scenarios_path} does not exist."
            "Make sure to run the script from the data directory or provide a base path."
        )

    if not (scenarios_path / "scenarios_gpkg").is_dir():
        raise ValueError(
            f"Scenarios path {scenarios_path / 'scenarios_gpkg'} does not exist."
            "Could not find base scenario data."
        )

    if not (scenarios_path / "scenarios_omx").is_dir():
        raise ValueError(
            f"Scenarios path {scenarios_path / 'scenarios_omx'} does not exist."
            "Could not find base scenario data."
        )

    if output_path.is_dir():
        raise FileExistsError(
            f"Output path {output_path} already exists."
            "Please remove it before running the assignment."
        )

    # create the output directory
    output_path.mkdir()
    output_reports_path.mkdir()

    # find all scenario files
    scenario_gpkgs = list((scenarios_path / "scenarios_gpkg").glob("scenario_*.gpkg"))
    scenario_omxs = list((scenarios_path / "scenarios_omx").glob("scenario_*.omx"))

    if len(scenario_gpkgs) != len(scenario_omxs):
        raise ValueError(
            f"Number of scenario gpkgs ({len(scenario_gpkgs)}) does not match number of scenario omxs ({len(scenario_omxs)})"
        )

    print(f"Found {len(scenario_gpkgs)} scenario files to process.")

    scenario_names = [scenario_gpkg.stem for scenario_gpkg in scenario_gpkgs]

    print(f"Starting assignment for {len(scenario_names)} scenario files...")

    # create partial function
    worker_func = partial(
        run_assignment,
        scenarios_dir=scenarios_path,
        output_dir=output_path,
        block_centroid_flows=CENTROID_FLOW_BLOCKING[network],
    )

    # run simulation
    results = [False] * len(scenario_names)
    i = 0
    for scenario in tqdm(scenario_names):
        results[i] = worker_func(scenario)
        i += 1

    print("--- Assignment Complete ---")
    success_count = sum(1 for res in results if res)
    print(results)
    print(f"Successfully ran {success_count} / {len(scenario_names)} static assignments.")
    print(f"Results saved to: {output_path}")
