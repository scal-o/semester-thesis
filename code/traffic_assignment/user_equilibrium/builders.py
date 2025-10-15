import json
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
import openmatrix as omx
import pandas as pd


class NetworkBuilder:
    """
    Builder for creating and serializing network scenarios from TNTP files.

    This class reads a TNTP network file once and generates multiple scenario
    variations by applying transformation functions to specific columns.
    """

    def __init__(
        self,
        network_file: Union[str, Path],
        skiprows: int = 8,
        scenarios_dir: Union[str, Path] = None,
    ):
        """
        Initialize the ScenarioBuilder.

        Reads the TNTP network file and creates the base network DataFrame.

        Parameters
        ----------
        network_file : Union[str, Path]
            Path to the TNTP network file (net.tntp)
        skiprows : int, optional
            Number of rows to skip when reading the file (default: 8)
        scenarios_dir : Union[str, Path], optional
            Directory to store scenario files (defaults to: network_file_dir /  "scenarios")
        """
        self.network_file = Path(network_file)
        self.skiprows = skiprows

        if not scenarios_dir:
            self.scenarios_dir = self.network_file.parent / "scenarios"
        else:
            self.scenarios_dir = Path(scenarios_dir)

        # Create scenarios directory
        self.scenarios_dir.mkdir(parents=False, exist_ok=True)

        # Read and store base network
        self.root = self._read_network()

        print(
            f"Loaded base network with {len(self.root)} links, "
            f"{len(pd.concat([self.root['a_node'], self.root['b_node']]).unique())} nodes"
        )

    def _read_network(self) -> pd.DataFrame:
        """
        Read the TNTP network file and process it into a DataFrame.

        Returns
        -------
        pd.DataFrame
            Processed base network DataFrame
        """
        # Read the network file
        net = pd.read_csv(self.network_file, skiprows=self.skiprows, sep="\t")

        # Adjust column names to standard format
        net.columns = [
            "newline",
            "a_node",
            "b_node",
            "capacity",
            "length",
            "free_flow_time",
            "b",
            "power",
            "speed",
            "toll",
            "link_type",
            "terminator",
        ]

        # Drop empty columns
        net.drop(columns=["newline", "terminator"], axis=1, inplace=True)

        # Add link id and direction column
        # direction = 1 indicates a one-way link from a to b
        net.insert(0, "link_id", np.arange(1, net.shape[0] + 1))
        net = net.assign(direction=1)

        return net

    def build_scenarios(
        self,
        column_name: str,
        transform_func: Callable[[pd.Series], pd.Series],
        num_scenarios: int,
        prefix: str = "scenario",
    ) -> List[str]:
        """
        Build multiple scenarios by applying a transformation function to a column.

        The function can include randomness to generate different scenarios.
        Each scenario is saved as a JSON file in the scenarios directory.

        Parameters
        ----------
        column_name : str
            Name of the column to transform (e.g., "capacity", "free_flow_time")
        transform_func : Callable[[pd.Series], pd.Series]
            Function that takes a pandas Series and returns a transformed Series.
            Can include randomness to generate different scenarios.
        num_scenarios : int
            Number of scenarios to generate
        prefix : str, optional
            Prefix for scenario filenames (default: "scenario")
            Files will be named as: {prefix}_001.json, {prefix}_002.json, etc.

        Returns
        -------
        List[str]
            List of created scenario filenames

        Examples
        --------
        >>> builder = ScenarioBuilder("net.tntp")
        >>>
        >>> # Generate 10 scenarios with random capacity reductions
        >>> def random_capacity(series):
        ...     return series * np.random.uniform(0.5, 0.9, len(series))
        >>>
        >>> filenames = builder.build_scenarios(
        ...     "capacity",
        ...     random_capacity,
        ...     num_scenarios=10,
        ...     prefix="random_cap"
        ... )
        """
        # Validate column exists
        if column_name not in self.root.columns:
            raise ValueError(
                f"Column '{column_name}' not found in network. "
                f"Available columns: {list(self.root.columns)}"
            )

        created_files = []

        print(f"Creating {num_scenarios} scenarios...")

        for i in range(1, num_scenarios + 1):
            # Create a copy of the base network
            scenario_df = self.root.copy()

            # Apply transformation to the specified column
            scenario_df[column_name] = transform_func(scenario_df[column_name])

            # Generate filename with zero-padded number
            scenario_name = f"{prefix}_{i:03d}"
            filepath = self.scenarios_dir / f"{scenario_name}.json"

            # Create metadata
            metadata = {
                "scenario_name": scenario_name,
                "base_network": str(self.network_file),
                "transformed_column": column_name,
                "num_links": len(scenario_df),
                "scenario_index": i,
                "total_scenarios": num_scenarios,
            }

            # Prepare data for serialization
            data = {"metadata": metadata, "network": scenario_df.to_dict(orient="records")}

            # Save to JSON
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            created_files.append(scenario_name)

            # Progress indicator
            if i % max(1, num_scenarios // 10) == 0 or i == num_scenarios:
                print(f"Progress: {i}/{num_scenarios} scenarios created")

        print(f"Created {num_scenarios} scenarios in {self.scenarios_dir}/")
        print(f"Files: {prefix}_001.json to {prefix}_{num_scenarios:03d}.json")

        return created_files

    def list_scenarios(self) -> pd.DataFrame:
        """
        List all scenarios in the scenarios directory.

        Returns
        -------
        pd.DataFrame
            DataFrame with scenario information from metadata
        """
        scenario_files = sorted(self.scenarios_dir.glob("*.json"))

        if not scenario_files:
            print(f"No scenarios found in {self.scenarios_dir}/")
            return pd.DataFrame(
                columns=[
                    "scenario_name",
                    "created_at",
                    "transformed_column",
                    "num_links",
                    "scenario_index",
                    "total_scenarios",
                ]
            )

        scenarios_info = []

        for filepath in scenario_files:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    scenarios_info.append(data["metadata"])
            except Exception as e:
                print(f"Warning: Could not read {filepath.name}: {e}")

        return pd.DataFrame(scenarios_info)

    def get_scenario(self, scenario_name: str) -> pd.DataFrame:
        """
        Load a specific scenario by name.

        Parameters
        ----------
        scenario_name : str
            Name of the scenario (without .json extension)

        Returns
        -------
        pd.DataFrame
            Network DataFrame for the scenario
        """
        filepath = self.scenarios_dir / f"{scenario_name}.json"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Scenario '{scenario_name}' not found. "
                f"Available scenarios:\n{self.list_scenarios()['scenario_name'].tolist()}"
            )

        with open(filepath, "r") as f:
            data = json.load(f)

        return pd.DataFrame(data["network"])

    def delete_scenario(self, scenario_name: str) -> None:
        """
        Delete a specific scenario file.

        Parameters
        ----------
        scenario_name : str
            Name of the scenario to delete (without .json extension)
        """
        filepath = self.scenarios_dir / f"{scenario_name}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Scenario '{scenario_name}' not found")

        filepath.unlink()
        print(f"Deleted scenario '{scenario_name}'")

    def clear_scenarios(self, confirm: bool = False) -> None:
        """
        Delete all scenario files.

        Parameters
        ----------
        confirm : bool
            Must be True to actually clear scenarios (safety measure)
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear all scenarios")

        scenario_files = list(self.scenarios_dir.glob("*.json"))

        for filepath in scenario_files:
            filepath.unlink()

        print(f"Deleted {len(scenario_files)} scenarios from {self.scenarios_dir}/")


class ODMatrixBuilder:
    """
    Builder for creating and serializing OD matrix scenarios from TNTP files.

    This class reads a TNTP trips file once and generates multiple OD matrix
    variations by applying transformation functions to the demand values.
    """

    def __init__(self, trips_file: Union[str, Path], matrices_dir: Union[str, Path] = None):
        """
        Initialize the ODMatrixBuilder.

        Reads the TNTP trips file and creates the base OD matrix.

        Parameters
        ----------
        trips_file : Union[str, Path]
            Path to the TNTP trips file (e.g., trips.tntp)
        matrices_dir : Union[str, Path], optional
            Directory to store OD matrix files (defaults to: trips_file_dir / "matrices")
        """
        self.trips_file = Path(trips_file)

        if not matrices_dir:
            self.matrices_dir = self.trips_file.parent / "matrices"
        else:
            self.matrices_dir = Path(matrices_dir)

        # Create matrices directory
        self.matrices_dir.mkdir(parents=True, exist_ok=True)

        # Read and store base OD matrix
        self.root, self.zones = self._read_trips_file()
        self.zone_index = np.arange(self.zones) + 1

        print(
            f"Loaded base OD matrix: {self.zones} zones, "
            f"{np.count_nonzero(self.root)} non-zero OD pairs, "
            f"total demand: {self.root.sum():.2f}"
        )

    def _read_trips_file(self) -> tuple[np.ndarray, int]:
        """
        Read the TNTP trips file and parse it into an OD matrix.

        Returns
        -------
        tuple[np.ndarray, int]
            OD matrix (numpy array) and number of zones
        """
        with open(self.trips_file, "r") as f:
            lines = f.read()
            blocks = lines.split("Origin")[1:]

        # Initialize matrix dictionary
        matrix_dict = {}

        # Iterate through all OD submatrices
        for block in blocks:
            orig = block.split("\n")
            orig = [line.strip() for line in orig]

            # Parse destinations
            dests = [dest for dests in orig[1:] for dest in dests.split(";")]
            dests = [line.strip() for line in dests if line.strip()]

            # Parse origin
            origin = int(orig[0])

            # Parse destination:value pairs
            destinations = {
                int(dest.split(":")[0].strip()): float(dest.split(":")[1].strip()) for dest in dests
            }

            matrix_dict[origin] = destinations

        # Determine number of zones
        zones = max(matrix_dict.keys())

        # Convert to numpy matrix
        mat = np.zeros((zones, zones))
        for i in range(1, zones + 1):
            for j in range(1, zones + 1):
                mat[i - 1, j - 1] = matrix_dict.get(i, {}).get(j, 0.0)

        return mat, zones

    def build_matrices(
        self,
        transform_func: Callable[[np.ndarray], np.ndarray],
        num_matrices: int,
        prefix: str = "od_matrix",
    ) -> List[str]:
        """
        Build multiple OD matrix scenarios by applying a transformation function.

        The function can include randomness to generate different scenarios.
        Each matrix is saved as an OMX file in the matrices directory.

        Parameters
        ----------
        transform_func : Callable[[np.ndarray], np.ndarray]
            Function that takes a numpy array (OD matrix) and returns a transformed array.
            Can include randomness to generate different scenarios.
        num_matrices : int
            Number of OD matrix scenarios to generate
        prefix : str, optional
            Prefix for matrix filenames (default: "od_matrix")
            Files will be named as: {prefix}_001.omx, {prefix}_002.omx, etc.

        Returns
        -------
        List[str]
            List of created matrix filenames (without .omx extension)

        Examples
        --------
        >>> builder = ODMatrixBuilder("trips.tntp")
        >>>
        >>> # Generate 10 matrices with random demand scaling
        >>> def random_demand(matrix):
        ...     scale = np.random.uniform(0.8, 1.2)
        ...     return matrix * scale
        >>>
        >>> filenames = builder.build_matrices(
        ...     random_demand,
        ...     num_matrices=10,
        ...     prefix="random_demand"
        ... )
        """
        created_files = []

        print(f"Creating {num_matrices} OD matrices...")

        for i in range(1, num_matrices + 1):
            # Create a copy of the base matrix
            matrix = self.root.copy()

            # Apply transformation
            matrix = transform_func(matrix)

            # Ensure no negative values
            matrix = np.maximum(matrix, 0)

            # Generate filename with zero-padded number
            matrix_name = f"{prefix}_{i:03d}"
            omx_filepath = self.matrices_dir / f"{matrix_name}.omx"
            json_filepath = self.matrices_dir / f"{matrix_name}_meta.json"

            # Create metadata
            metadata = {
                "matrix_name": matrix_name,
                "base_trips_file": str(self.trips_file),
                "num_zones": int(self.zones),
                "total_demand": float(matrix.sum()),
                "non_zero_pairs": int(np.count_nonzero(matrix)),
                "matrix_index": i,
                "total_matrices": num_matrices,
            }

            # Save OMX file
            with omx.open_file(str(omx_filepath), "w") as omx_file:
                omx_file["matrix"] = matrix
                omx_file.create_mapping("taz", self.zone_index)

            # Save metadata as JSON
            with open(json_filepath, "w") as f:
                json.dump(metadata, f, indent=2)

            created_files.append(matrix_name)

            # Progress indicator
            if i % max(1, num_matrices // 10) == 0 or i == num_matrices:
                print(f"Progress: {i}/{num_matrices} matrices created")

        print(f"Created {num_matrices} OD matrices in {self.matrices_dir}/")
        print(f"Files: {prefix}_001.omx to {prefix}_{num_matrices:03d}.omx")

        return created_files

    def list_matrices(self) -> list[dict]:
        """
        List all OD matrices in the matrices directory.

        Returns
        -------
        list[dict]
            List of dictionaries containing matrix metadata
        """
        metadata_files = sorted(self.matrices_dir.glob("*_meta.json"))

        if not metadata_files:
            print(f"No OD matrices found in {self.matrices_dir}/")
            return []

        matrices_info = []

        for filepath in metadata_files:
            try:
                with open(filepath, "r") as f:
                    metadata = json.load(f)
                    matrices_info.append(metadata)
            except Exception as e:
                print(f"Warning: Could not read {filepath.name}: {e}")

        return matrices_info

    def get_matrix(self, matrix_name: str) -> tuple[np.ndarray, dict]:
        """
        Load a specific OD matrix by name.

        Parameters
        ----------
        matrix_name : str
            Name of the matrix (without .omx extension)

        Returns
        -------
        tuple[np.ndarray, dict]
            OD matrix (numpy array) and metadata dictionary
        """
        omx_filepath = self.matrices_dir / f"{matrix_name}.omx"
        json_filepath = self.matrices_dir / f"{matrix_name}_meta.json"

        if not omx_filepath.exists():
            available = [m["matrix_name"] for m in self.list_matrices()]
            raise FileNotFoundError(
                f"Matrix '{matrix_name}' not found. Available matrices: {available}"
            )

        # Load OMX file
        with omx.open_file(str(omx_filepath), "r") as omx_file:
            matrix = np.array(omx_file["matrix"])

        # Load metadata
        with open(json_filepath, "r") as f:
            metadata = json.load(f)

        return matrix, metadata

    def delete_matrix(self, matrix_name: str) -> None:
        """
        Delete a specific OD matrix and its metadata.

        Parameters
        ----------
        matrix_name : str
            Name of the matrix to delete (without .omx extension)
        """
        omx_filepath = self.matrices_dir / f"{matrix_name}.omx"
        json_filepath = self.matrices_dir / f"{matrix_name}_meta.json"

        if not omx_filepath.exists():
            raise FileNotFoundError(f"Matrix '{matrix_name}' not found")

        omx_filepath.unlink()
        if json_filepath.exists():
            json_filepath.unlink()

        print(f"✓ Deleted matrix '{matrix_name}'")

    def clear_matrices(self, confirm: bool = False) -> None:
        """
        Delete all OD matrix files.

        Parameters
        ----------
        confirm : bool
            Must be True to actually clear matrices (safety measure)
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear all matrices")

        omx_files = list(self.matrices_dir.glob("*.omx"))
        json_files = list(self.matrices_dir.glob("*_meta.json"))

        for filepath in omx_files + json_files:
            filepath.unlink()

        print(f"✓ Deleted {len(omx_files)} matrices from {self.matrices_dir}/")

    def get_summary(self) -> dict:
        """
        Get summary statistics of all stored matrices.

        Returns
        -------
        dict
            Summary statistics including counts and demand ranges
        """
        matrices = self.list_matrices()

        if not matrices:
            return {
                "num_matrices": 0,
                "total_zones": self.zones,
                "base_demand": float(self.root.sum()),
            }

        demands = [m["total_demand"] for m in matrices]

        return {
            "num_matrices": len(matrices),
            "total_zones": self.zones,
            "base_demand": float(self.root.sum()),
            "scenario_demands": {
                "min": min(demands),
                "max": max(demands),
                "mean": np.mean(demands),
                "std": np.std(demands),
            },
        }
