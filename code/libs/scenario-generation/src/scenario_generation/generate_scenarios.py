"""
Script for Scenario Generation

This script reads a "_master" GeoPackage file for a network and
generates N scenario .gpkg files using a ScenarioGenerator class.
"""

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Dict, Tuple

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Modification Function Definitions ---
# These functions are "strategies" and can remain at the module level.
# -----------------------------------------------------------------


def modify_capacity_uniform(x: pd.Series) -> pd.Series:
    """
    Applies a stochastic modification factor U(0.8, 1.0) to capacity.

    Args:
        x: A pd.Series containing the original capacity values.

    Returns:
        A pd.Series with the modified and rounded capacity values.
    """
    factors = np.random.uniform(0.8, 1.0, size=len(x))
    return (x * factors).round(0).astype(int)


def modify_fft_normal(x: pd.Series) -> pd.Series:
    """
    Applies a stochastic modification factor N(1.0, 0.1) to free-flow time.

    Args:
        x: A pd.Series containing the original free_flow_time values.

    Returns:
        A pd.Series with the modified free_flow_time values.
    """
    factors = np.random.normal(1.0, 0.1, size=len(x))
    factors_clipped = np.clip(factors, 0.5, 1.5)
    return x * factors_clipped


def modify_od_uniform(x: pd.Series) -> pd.Series:
    """
    Applies a stochastic modification factor U(0.5, 1.5) to demand.

    Args:
        x: A pd.Series containing the original demand values.

    Returns:
        A pd.Series with the modified and rounded demand values.
    """
    factors = np.random.uniform(0.5, 1.5, size=len(x))
    return (x * factors).round(0).astype(int)


# --- Modification Rules Mapping ---
# This dictionary "maps" attributes to their modification functions.
# -----------------------------------------------------------------
MODIFICATION_RULES: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "capacity": modify_capacity_uniform,
    # "free_flow_time": modify_fft_normal,
    "demand": modify_od_uniform,
}


# --- Scenario Generator Class Definition ---
# This class encapsulates the scenario generation logic.
# -----------------------------------------------------------------
class ScenarioGenerator:
    """
    Loads a master network and generates stochastic scenarios.

    Attributes:
        base_nodes_gdf (gpd.GeoDataFrame): The master (unmodified) node data.
        base_links_gdf (gpd.GeoDataFrame): The master (unmodified) link data.
        base_od_gdf (gpd.GeoDataFrame): The master (unmodified) OD data.
    """

    def __init__(self, network: str, path: Path):
        """
        Initializes the generator by loading the master network data.

        Args:
            network: The name of the network (e.g., 'anaheim').
            path: The base path to the networks directory.

        Raises:
            FileNotFoundError: If the master .gpkg file is not found.
        """
        self.network_name: str = network
        self.network_path: Path = path / network
        self.master_gpkg_file: Path = self.network_path / f"{self.network_name}_master.gpkg"

        print(f"Loading master network from: {self.master_gpkg_file}")
        self._load_master_data()
        print(
            f"Loaded:\n"
            f"- {len(self.base_nodes_gdf)} master nodes\n"
            f"- {len(self.base_links_gdf)} master links\n"
            f"- {len(self.base_od_gdf)} master OD pairs\n"
            f"- {len(self.base_flows_gdf)} master flows\n"
        )

    def _load_master_data(self) -> None:
        """
        Internal helper to load nodes and links from the master .gpkg.
        """
        if not self.master_gpkg_file.exists():
            raise FileNotFoundError(f"Master file not found: {self.master_gpkg_file}")
        try:
            self.base_nodes_gdf: gpd.GeoDataFrame = gpd.read_file(
                self.master_gpkg_file, layer="nodes"
            )
            self.base_links_gdf: gpd.GeoDataFrame = gpd.read_file(
                self.master_gpkg_file, layer="links"
            )
            self.base_od_gdf: gpd.GeoDataFrame = gpd.read_file(self.master_gpkg_file, layer="od")
            self.base_flows_gdf: gpd.GeoDataFrame = gpd.read_file(
                self.master_gpkg_file, layer="flows"
            )

            self.base_flows_gdf = gpd.GeoDataFrame(self.base_flows_gdf, geometry=None)

        except Exception as e:
            raise Exception(
                "Failed to read layers from master GeoPackage."
                "Ensure the file contains 'nodes', 'links', and 'od' layers."
            ) from e

    @staticmethod
    def _generate_single_scenario(
        scenario_seed: int,
        base_nodes_gdf: gpd.GeoDataFrame,
        base_links_gdf: gpd.GeoDataFrame,
        base_od_gdf: gpd.GeoDataFrame,
        base_flows_gdf: gpd.GeoDataFrame,
        output_dir: Path,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Generates a single scenario by applying modifications to its link and demand layers.
        Static method meant to be used with multiprocessor for parallelization.

        Args:
            scenario_seed: The ID (seed) for this scenario.
            base_nodes_gdf: The master (unmodified) node data.
            base_links_gdf: The master (unmodified) link data.
            base_od_gdf: The master (unmodified) OD data.
            output_dir: The directory to save the scenario .gpkg file.

        """
        # Set the seed for reproducibility
        np.random.seed(scenario_seed)

        # Work on a copy
        mod_links_gdf = base_links_gdf.copy()
        mod_od_gdf = base_od_gdf.copy()

        # If the scenario seed is 0, do not apply any modifications but simply
        # convert the base data to the correct format
        if scenario_seed != 0:
            # Apply modification to the od matrix
            if "demand" in MODIFICATION_RULES:
                mod_function = MODIFICATION_RULES.pop("demand")
                original_series = mod_od_gdf["demand"]
                mod_od_gdf["demand"] = mod_function(original_series)

            # Apply modifications to the links
            for attribute, mod_function in MODIFICATION_RULES.items():
                if attribute not in mod_links_gdf.columns:
                    print(f"  Warning: Attribute '{attribute}' not found. Skipping.")
                    continue

                original_series = mod_links_gdf[attribute]
                mod_links_gdf[attribute] = mod_function(original_series)

        # Save scenario files
        scenario_filename = f"scenario_{scenario_seed:05d}"
        scenario_output_dir = output_dir / scenario_filename
        scenario_output_dir.mkdir(parents=True, exist_ok=False)

        try:
            base_nodes_gdf.to_file(scenario_output_dir / "nodes.geojson", driver="GeoJSON")
            mod_links_gdf.to_file(scenario_output_dir / "links.geojson", driver="GeoJSON")
            mod_od_gdf.to_file(scenario_output_dir / "od.geojson", driver="GeoJSON")

            if scenario_seed == 0:
                base_flows_gdf.to_file(scenario_output_dir / "flows.geojson", driver="GeoJSON")

        except Exception as e:
            raise Exception(f"Error while saving {scenario_filename}: {e}")

        return True

    def run(self, n_scenarios: int, output_dir: Path, processes: int = None) -> None:
        """
        Runs the full scenario generation process.

        This method loops N times, generates each scenario, and saves
        it to a separate .gpkg file.

        Args:
            n_scenarios: The total number of scenarios to generate.
            output_dir: The directory to save the .gpkg files.
        """
        print(f"\nStarting generation of {n_scenarios} scenarios...")
        print(f"Modification rules: {list(MODIFICATION_RULES.keys())}")

        output_dir.mkdir(parents=True, exist_ok=False)

        # Create a partial function to pass fixed arguments
        worker_func = partial(
            ScenarioGenerator._generate_single_scenario,
            base_nodes_gdf=self.base_nodes_gdf,
            base_links_gdf=self.base_links_gdf,
            base_od_gdf=self.base_od_gdf,
            base_flows_gdf=self.base_flows_gdf,
            output_dir=output_dir,
        )

        # Generate scenario IDs
        scenario_ids = range(0, n_scenarios + 1)

        # Create a process pool
        if processes is None:
            processes = cpu_count()

        with Pool(processes=processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_func, scenario_ids),
                    total=n_scenarios,
                    desc="Generating Scenarios",
                )
            )

        print("--- Scenario Generation Complete ---")
        print(f"Successfylly wrote {len(results)} scenarios to {output_dir}")


# --- CLI Definition ---
# This section defines the command-line interface using Click.
# -----------------------------------------------------------------
@click.command("generate")
@click.argument("network")
@click.option(
    "--path",
    default="networks",
    show_default=True,
    help="The base path to the networks directory.",
)
@click.option(
    "--output",
    default="data/",
    show_default=True,
    help="Directory to save the generated scenario .gpkg files.",
)
@click.option(
    "-n",
    "--n-scenarios",
    type=int,
    default=5000,
    show_default=True,
    help="Number of scenarios to generate.",
)
@click.option(
    "-p",
    "--processes",
    type=int,
    default=None,
    help="Number of parallel processes to use. Defaults to number of CPU cores.",
)
def generate_scenarios(
    network: str,
    path: str = None,
    output: str = None,
    n_scenarios: int = 5000,
    processes: int = None,
):
    """
    Generates N stochastic scenarios from a NETWORK's master GeoPackage file.

    This script instantiates and runs the ScenarioGenerator class.

    The applied modifications are defined in a local MODIFICATION_RULES dictionary.
    """
    print("--- Scenario Generation ---")
    print(f"Network: {network}")
    print(f"Base path: {path}")
    print(f"Output directory: {output}")

    # setting up paths
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    network_path = path / network

    if output is None:
        output_path = Path.cwd() / "data" / network / "scenarios_geojson"
    else:
        output_path = Path(output) / network / "scenarios_geojson"

    if not network_path.is_dir():
        raise ValueError(
            f"Network path {path} does not exist."
            "Make sure to run the script from the network directory or provide a base_path."
        )

    if output_path.exists():
        raise FileExistsError(
            f"Output directory {output_path} already exists."
            "Please remove it or choose a different output path."
        )

    try:
        # 1. Initialize the generator (this loads the data)
        generator = ScenarioGenerator(network, path)

        # 2. Run the generation process
        generator.run(n_scenarios, output_path, processes)

    except Exception as e:
        raise Exception(f"Scenario generation failed. An unexpected error occurred: {e}") from e
