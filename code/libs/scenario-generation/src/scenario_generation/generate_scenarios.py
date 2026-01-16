"""
Script for Scenario Generation

This script reads a "_master" GeoPackage file for a network and
generates N scenario .gpkg files using a ScenarioGenerator class.
"""

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, Optional

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import tqdm

# --- Modification Function Definitions ---
# These functions are "strategies" and can remain at the module level.
# -----------------------------------------------------------------


def modify_capacity_lhs(x: pd.Series) -> pd.Series:
    """
    Applies a stochastic modification factor using Latin Hypercube Sampling to capacity.

    Args:
        x: A pd.Series containing the original capacity values.

    Returns:
        A pd.Series with the modified and rounded capacity values.
    """
    # use the default capacity LHS sampler
    return DEFAULT_CAPACITY_LHS_SAMPLER(x)


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
    return np.ceil(x * factors).astype(int)


class LatinHypercubeSampler:
    """
    Latin Hypercube Sampler that precomputes a matrix of weights and returns
    a row of weights each time it is called.

    Usage:
    sampler = LatinHypercubeSampler(value_range=(0.8, 1.0), sample_size=80, num_samples=5000)
    weights = sampler(pd.Series(...))  # returns a pd.Series with appropriate dtype
    """

    def __init__(
        self,
        value_range: tuple[float, float],
        sample_size: int,
        num_samples: int = 5000,
    ) -> None:
        self.low = float(value_range[0])
        self.high = float(value_range[1])
        self.num_samples = int(num_samples)
        # number of columns (per-sample size)
        self._n_rows = self.num_samples
        self._sample_size = sample_size
        self._matrix: Optional[np.ndarray] = None
        self._index = 0
        # RNG is no longer used; SciPy's qmc handles its internal RNG
        # If a sample_size is provided, build the matrix immediately
        if self._sample_size is not None:
            self._build_matrix(self._sample_size)

    def _build_matrix(self, sample_size: int) -> None:
        """
        Build an LHS matrix of shape (n_rows, sample_size).
        The matrix is generated column-wise, shuffling each column independently.
        """
        n_rows = self._n_rows
        # Each column is a LHS of size n_rows
        mat = np.empty((n_rows, sample_size), dtype=float)

        # Use SciPy's LHS generator to build the full matrix at once
        lhs_sampler = qmc.LatinHypercube(d=sample_size)
        u = lhs_sampler.random(n_rows)
        # scale to [low, high] for each dimension
        mat = qmc.scale(u, [self.low] * sample_size, [self.high] * sample_size)

        self._matrix = mat

    def __call__(self, x: pd.Series) -> pd.Series:
        """Return a new vector of sampled weights multiplied with `x`.

        The method will lazily initialize the internal LHS matrix if not already
        created, using the length of `x` as the sample size.
        It advances an internal index and returns the row at that index. The
        index wraps around once it reaches the end of the matrix.
        """
        if self._matrix is None or self._sample_size != len(x):
            # initialize or re-initialize matrix based on incoming size
            self._build_matrix(len(x))

        assert self._matrix is not None

        weights = self._matrix[self._index, :]
        self._index += 1
        if self._index >= self._n_rows:
            self._index = 0

        # apply weights
        result = x * weights
        # if integer-like, round and cast
        if pd.api.types.is_integer_dtype(x.dtype):
            return result.round(0).astype(int)
        return result


# Default capacity sampler: 80 weights per scenario by default.
# Change num_samples or sample_size here if you want a different behavior.
DEFAULT_CAPACITY_LHS_SAMPLER = LatinHypercubeSampler(
    value_range=(0.8, 1.0), num_samples=5000, sample_size=80
)


# --- Modification Rules Mapping ---
# This dictionary "maps" attributes to their modification functions.
# -----------------------------------------------------------------
MODIFICATION_RULES: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "capacity": modify_capacity_lhs,
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
        capacity_sampler: "LatinHypercubeSampler | None" = None,
    ) -> bool:
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
            # Use a copy of the rules to avoid mutating the global dictionary
            rules = MODIFICATION_RULES.copy()
            # Demand modification is handled separately
            demand_mod_function = rules.pop("demand")
            if demand_mod_function is not None:
                original_series = mod_od_gdf["demand"]
                mod_od_gdf["demand"] = demand_mod_function(original_series)

            # Apply modifications to the links
            for attribute, mod_function in rules.items():
                if attribute not in mod_links_gdf.columns:
                    print(f"  Warning: Attribute '{attribute}' not found. Skipping.")
                    continue

                original_series = mod_links_gdf[attribute]
                # If we have a capacity sampler and the rule is the LHS wrapper
                if (
                    attribute == "capacity"
                    and mod_function is modify_capacity_lhs
                    and capacity_sampler is not None
                ):
                    mod_links_gdf[attribute] = capacity_sampler(original_series)
                else:
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

    def run(self, n_scenarios: int, output_dir: Path, multiprocess: bool = False) -> None:
        """
        Runs the full scenario generation process.

        This method loops N times, generates each scenario, and saves
        it to a separate .gpkg file.

        Args:
            n_scenarios: The total number of scenarios to generate.
            output_dir: The directory to save the .gpkg files.
            multiprocess: Whether to use multiprocessing for parallel execution (max 10 processes).
        """
        print(f"\nStarting generation of {n_scenarios} scenarios...")
        print(f"Modification rules: {list(MODIFICATION_RULES.keys())}")

        output_dir.mkdir(parents=True, exist_ok=False)

        # Create a partial function to pass fixed arguments
        # Prepare a capacity LHS sampler if capacity uses LHS modifier
        capacity_sampler = None
        if (
            "capacity" in MODIFICATION_RULES
            and MODIFICATION_RULES["capacity"] is modify_capacity_lhs
        ):
            # `num_samples` must match the number of scenarios to be generated.
            # We include `scenario_00000` as well, so use n_scenarios + 1.
            capacity_sampler = LatinHypercubeSampler(
                value_range=(0.5, 1.0),
                sample_size=len(self.base_links_gdf),
                num_samples=(n_scenarios + 1),
            )

        worker_func = partial(
            ScenarioGenerator._generate_single_scenario,
            base_nodes_gdf=self.base_nodes_gdf,
            base_links_gdf=self.base_links_gdf,
            base_od_gdf=self.base_od_gdf,
            base_flows_gdf=self.base_flows_gdf,
            output_dir=output_dir,
            capacity_sampler=capacity_sampler,
        )

        # Generate scenario IDs
        scenario_ids = list(range(0, n_scenarios + 1))

        # Run with or without multiprocessing
        if multiprocess:
            num_processes = cpu_count()
            num_processes = min(num_processes - 2, len(scenario_ids))
            print(f"Using {num_processes} parallel processes...")

            with Pool(processes=num_processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(worker_func, scenario_ids),
                        total=len(scenario_ids),
                        desc="Generating Scenarios",
                    )
                )
        else:
            print("Running sequentially (single process)...")
            results = []
            for scenario_id in tqdm(scenario_ids, total=len(scenario_ids), desc="Generating Scenarios"):
                result = worker_func(scenario_id)
                results.append(result)

        print("--- Scenario Generation Complete ---")
        print(f"Successfully wrote {len(results)} scenarios to {output_dir}")


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
    "--multiprocess",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use multiprocessing for parallel scenario generation.",
)
def generate_scenarios(
    network: str,
    path: Optional[str] = None,
    output: Optional[str] = None,
    n_scenarios: int = 5000,
    multiprocess: bool = False,
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
    print(f"Multiprocessing: {multiprocess}")

    # setting up paths
    if path is None:
        path_path = Path.cwd()
    else:
        path_path = Path(path)

    network_path = path_path / network

    if output is None:
        output_path = Path.cwd() / "data" / network / "scenarios_geojson"
    else:
        output_path = Path(output) / "scenarios_geojson"

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
        generator = ScenarioGenerator(network, path_path)

        # 2. Run the generation process
        generator.run(n_scenarios, output_path, multiprocess)

    except Exception as e:
        raise Exception(f"Scenario generation failed. An unexpected error occurred: {e}") from e
