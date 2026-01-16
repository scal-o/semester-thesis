import io
import logging
from contextlib import redirect_stderr
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.traffic_assignment import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
from tqdm import tqdm

logger = logging.getLogger("aequilibrae")
logger.setLevel(logging.ERROR)

CENTROID_FLOW_BLOCKING = {"anaheim": True, "sioux_falls": False, "chicago": False}


def find_centroids(scenario_path: Path):
    """
    Small function to find centroids for a given scenario (from OD matrix).
    """
    # load scenario data
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario '{scenario_path.name}' data not found. ")
    elif not (scenario_path / "nodes.geojson").exists():
        raise FileNotFoundError(f"Demand for scenario '{scenario_path.name}' not found. ")

    od_df = gpd.read_file(scenario_path / "od.geojson")

    centroids = pd.concat([od_df["origin"], od_df["destination"]])
    centroids = centroids.unique()
    return centroids


def load_network(scenario_path: Path, block_centroid_flows: False):
    """
    Loader for network graphs.

    Loads graphs from scenario geojson files.
    """

    # load scenario data
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario '{scenario_path.name}' data not found. ")
    elif not (scenario_path / "links.geojson").exists():
        raise FileNotFoundError(f"Network for scenario '{scenario_path.name}' not found. ")
    elif not (scenario_path / "nodes.geojson").exists():
        raise FileNotFoundError(f"Nodes for scenario '{scenario_path.name}' not found. ")

    # load network centroids
    centroids = find_centroids(scenario_path)

    # load network data
    net_df = gpd.read_file(scenario_path / "links.geojson")
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
    nodes = gpd.read_file(scenario_path / "nodes.geojson")
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


def load_matrix(scenario_path: Path):
    """
    Loader for OD matrices.

    Loads matrices in AequilibraE matrix objects from OMX files.
    """

    # load scenario data
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario '{scenario_path.name}' data not found. ")
    elif not (scenario_path / "od.omx").exists():
        raise FileNotFoundError(f"Demand matrix for scenario '{scenario_path.name}' not found. ")

    # create AequilibraE matrix from OMX
    mat = AequilibraeMatrix()
    mat.create_from_omx(str(scenario_path / "od.omx"))
    mat.computational_view(["matrix"])

    return mat


def run_assignment(scenario_name: str, scenarios_dir: Path, block_centroid_flows=False):
    """
    Loader for a complete Assignment problem.

    Loads network and od matrix from a scenario and creates a TrafficAssignment instance.
    """

    # load dirs
    scenario_path = scenarios_dir / "scenarios_geojson" / f"{scenario_name}"
    output_path = scenarios_dir / "scenarios_sta_results" / f"{scenario_name}"
    output_path.mkdir(parents=False, exist_ok=False)

    # load graph and od matrix
    graph = load_network(scenario_path, block_centroid_flows)
    matrix = load_matrix(scenario_path)

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
    scenario_results_path = output_path / f"{scenario_name}.parquet"
    scenario_convergence_path = output_path / f"{scenario_name}_convergence.parquet"

    network.to_parquet(scenario_results_path)
    assignment.report().to_parquet(scenario_convergence_path)

    return True


@click.command("run")
@click.argument("network")
@click.option(
    "--path",
    default="data",
    show_default=True,
    help="The base path to the scenarios directory.",
)
def run_sta(
    network: str,
    path: str,
):
    """
    Runs static assignment using AequilibraE for all of NETWORK's scenarios.
    """
    print("--- Running static traffic assignment ---")
    print(f"Network: {network}")
    print(f"Path: {path}")

    # setting up paths
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    scenarios_path = path / network
    output_path = scenarios_path / "scenarios_sta_results"

    if "sioux_falls" in network:
        network = "sioux_falls"
    elif "anaheim" in network:
        network = "anaheim"
    elif "chicago" in network:
        network = "chicago"
    else:
        network = None

    # check paths
    if not scenarios_path.is_dir():
        raise ValueError(
            f"Scenarios path {scenarios_path} does not exist."
            "Make sure to run the script from the data directory or provide a base path."
        )

    if not (scenarios_path / "scenarios_geojson").is_dir():
        raise ValueError(
            f"Scenarios path {scenarios_path / 'scenarios_geojson'} does not exist."
            "Could not find base scenario data."
        )

    if output_path.is_dir():
        raise FileExistsError(
            f"Output path {output_path} already exists.\n"
            "Please remove it before running the assignment."
        )

    # create the output directory
    output_path.mkdir()

    # find all scenario files
    scenario_dirs = list((scenarios_path / "scenarios_geojson").glob("scenario_*"))

    print(f"Found {len(scenario_dirs)} scenario files to process.")

    scenario_names = [scenario.name for scenario in scenario_dirs]

    print(f"Starting assignment for {len(scenario_names)} scenario files...")

    # create partial function
    worker_func = partial(
        run_assignment,
        scenarios_dir=scenarios_path,
        block_centroid_flows=CENTROID_FLOW_BLOCKING.get(network, False),
    )

    # run simulation in parallel with max processes = num cores - 2
    num_processes = cpu_count() - 2
    num_processes = min(num_processes, len(scenario_names))
    print(f"Using {num_processes} parallel processes...")

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(worker_func, scenario_names),
                total=len(scenario_names),
                desc="Running STA & Writing results",
            )
        )

    print("--- Assignment Complete ---")
    success_count = sum(1 for res in results if res)
    print(results)
    print(f"Successfully ran {success_count} / {len(scenario_names)} static assignments.")
    print(f"Results saved to: {output_path}")
