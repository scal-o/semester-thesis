import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph


class GraphLoader:
    """
    Loader for network scenario graphs.

    Loads graphs on-demand from scenario JSON files. Caching is explicit
    via the preload() method.
    """

    def __init__(
        self,
        scenarios_dir: Union[str, Path] = "scenarios",
        cost_field: str = "free_flow_time",
        skim_fields: Optional[List[str]] = None,
        block_centroid_flows: bool = False,
    ):
        """
        Initialize GraphLoader.

        Parameters
        ----------
        scenarios_dir : Union[str, Path]
            Path to scenarios directory
        cost_field : str, optional
            Default cost field for graphs (default: "free_flow_time")
        skim_fields : List[str], optional
            Default skim fields (default: ["free_flow_time"])
        block_centroid_flows : bool, optional
            Whether to block centroid flows (default: False)
        """
        self.scenarios_dir = Path(scenarios_dir)
        self.cost_field = cost_field
        self.skim_fields = skim_fields if skim_fields is not None else ["free_flow_time"]
        self.block_centroid_flows = block_centroid_flows

        self._cache: Dict[tuple, Graph] = {}  # (scenario_name, centroids_hash) -> Graph

    def load(
        self,
        scenario_name: str,
        centroids: np.ndarray,
        cost_field: Optional[str] = None,
        skim_fields: Optional[List[str]] = None,
        block_centroid_flows: Optional[bool] = None,
    ) -> Graph:
        """
        Load a graph for a specific scenario and set of centroids.

        Parameters
        ----------
        scenario_name : str
            Name of the scenario (without .json extension)
        centroids : np.ndarray
            Array of centroid zone IDs
        cost_field : str, optional
            Override default cost field
        skim_fields : List[str], optional
            Override default skim fields
        block_centroid_flows : bool, optional
            Override default centroid flow blocking

        Returns
        -------
        Graph
            Configured AequilibraE Graph object
        """
        # Create cache key
        centroids_hash = hash(centroids.tobytes())
        cache_key = (scenario_name, centroids_hash)

        # Return from cache if available
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Use defaults if not overridden
        cost_field = cost_field if cost_field else self.cost_field
        skim_fields = skim_fields if skim_fields else self.skim_fields
        block_centroid_flows = (
            block_centroid_flows if block_centroid_flows else self.block_centroid_flows
        )

        # Load scenario data
        scenario_path = self.scenarios_dir / f"{scenario_name}.json"

        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario '{scenario_name}' not found. "
                f"Available scenarios: {self.list_scenarios()}"
            )

        with open(scenario_path, "r") as f:
            data = json.load(f)

        # Convert network to DataFrame
        net_df = pd.DataFrame(data["network"])

        # Create and configure graph
        graph = Graph()
        graph.network = net_df
        graph.prepare_graph(centroids)

        graph.set_graph(cost_field)
        graph.capacity = net_df["capacity"].values
        graph.free_flow_time = net_df["free_flow_time"].values

        graph.set_skimming(skim_fields)
        graph.set_blocked_centroid_flows(block_centroid_flows)
        graph.network["id"] = graph.network["link_id"]

        return graph

    def preload(
        self, scenario_names: Union[List[str], str], centroids: np.ndarray, **graph_params
    ) -> None:
        """
        Preload graphs into cache.

        Useful for batch operations to avoid repeated loading.

        Parameters
        ----------
        scenario_names : Union[List[str], str]
            List of scenario names or "all"
        centroids : np.ndarray
            Array of centroid zone IDs
        **graph_params
            Additional parameters passed to load()
        """
        if scenario_names == "all":
            scenario_names = self.list_scenarios()

        # check centroids shape
        # if centroids is a matrix, there must be an entry for each graph
        # if it's a vector, we can use the same centroids for every graph
        centroid_as_list = False
        if len(centroids.shape) == 2:
            if centroids.shape[0] == len(scenario_names):
                centroid_as_list = True
            elif centroids.shape[0] == 1:
                pass
            else:
                raise ValueError(
                    "The number of centroid vectors is not aligned with the number of graphs to be created."
                )

        print(f"Preloading {len(scenario_names)} graphs...")

        for i, scenario_name in enumerate(scenario_names):
            # Load centroid vector
            if centroid_as_list:
                centroid_vec = centroids[i]
            else:
                # if there is just one centroid vector, always return the first row of the matrix
                centroid_vec = centroids[0]

            # Create cache key
            centroids_hash = hash(centroid_vec.tobytes())
            cache_key = (scenario_name, centroids_hash)

            # Only load if not already cached
            if cache_key not in self._cache:
                graph = self.load(scenario_name, centroid_vec, **graph_params)
                self._cache[cache_key] = graph

            if i % max(1, len(scenario_names) // 10) == 0 or i == len(scenario_names):
                print(f"  Progress: {i}/{len(scenario_names)} graphs loaded")

        print(f"✓ Preloaded {len(scenario_names)} graphs")

    def list_scenarios(self) -> List[str]:
        """
        List all available scenario names.

        Returns
        -------
        List[str]
            List of scenario names (without .json extension)
        """
        return sorted([f.stem for f in self.scenarios_dir.glob("*.json")])

    def clear_cache(self) -> None:
        """Clear all cached graphs from memory."""
        num_cached = len(self._cache)
        self._cache.clear()
        print(f"✓ Cleared {num_cached} graphs from cache")

    @property
    def num_cached(self) -> int:
        """Number of currently cached graphs."""
        return len(self._cache)

    def __contains__(self, scenario_name: str) -> bool:
        """Check if a scenario exists."""
        return (self.scenarios_dir / f"{scenario_name}.json").exists()

    def __len__(self) -> int:
        """Number of available scenarios."""
        return len(self.list_scenarios())

    def __repr__(self) -> str:
        return (
            f"GraphLoader(scenarios_dir='{self.scenarios_dir}', "
            f"available={len(self)}, cached={self.num_cached})"
        )


class ODMatrixLoader:
    """
    Loader for OD matrices.

    Loads matrices on-demand from OMX files. Caching is explicit
    via the preload() method.
    """

    def __init__(self, matrices_dir: Union[str, Path] = "matrices", matrix_name: str = "matrix"):
        """
        Initialize MatrixLoader.

        Parameters
        ----------
        matrices_dir : Union[str, Path]
            Path to matrices directory
        matrix_name : str, optional
            Name of the matrix in OMX files (default: "matrix")
        """
        self.matrices_dir = Path(matrices_dir)
        self.matrix_name = matrix_name
        self._cache: Dict[str, AequilibraeMatrix] = {}  # matrix_name -> AequilibraeMatrix

    def load(self, matrix_name: str) -> AequilibraeMatrix:
        """
        Load an OD matrix.

        Parameters
        ----------
        matrix_name : str
            Name of the matrix (without .omx extension)

        Returns
        -------
        AequilibraeMatrix
            Loaded and configured AequilibraE matrix
        """
        # Return from cache if available
        if matrix_name in self._cache:
            return self._cache[matrix_name]

        # Load from file
        omx_path = self.matrices_dir / f"{matrix_name}.omx"

        if not omx_path.exists():
            raise FileNotFoundError(
                f"Matrix '{matrix_name}' not found. Available matrices: {self.list_matrices()}"
            )

        # Create AequilibraE matrix from OMX
        mat = AequilibraeMatrix()
        mat.create_from_omx(str(omx_path))
        mat.computational_view([self.matrix_name])

        return mat

    def get_centroids(self, matrix_names: list) -> np.ndarray:
        """
        Extract active centroid zone IDs from a matrix.

        Returns zones that have at least one trip as origin or destination.

        Parameters
        ----------
        matrix_name : str
            Name of the matrix

        Returns
        -------
        np.ndarray
            Array of active centroid zone IDs (1-based)
        """

        centroids = []

        for matrix_name in matrix_names:
            matrix = self.load(matrix_name)
            mat_array = matrix.matrix_view

            # Find zones with at least one trip (as origin OR destination)
            has_trips = (mat_array.sum(axis=0) > 0) | (mat_array.sum(axis=1) > 0)
            zone_indices = np.array([np.where(has_trips)[0]])

            # Convert to 1-based zone IDs
            centroids.append(zone_indices + 1)

        return np.concatenate(centroids)

    def preload(self, matrix_names: Union[List[str], str]) -> None:
        """
        Preload matrices into cache.

        Useful for batch operations to avoid repeated loading.

        Parameters
        ----------
        matrix_names : Union[List[str], str]
            List of matrix names or "all"
        """
        if matrix_names == "all":
            matrix_names = self.list_matrices()

        print(f"Preloading {len(matrix_names)} matrices...")

        for i, matrix_name in enumerate(matrix_names, 1):
            # Only load if not already cached
            if matrix_name not in self._cache:
                mat = self.load(matrix_name)
                self._cache[matrix_name] = mat

            if i % max(1, len(matrix_names) // 10) == 0 or i == len(matrix_names):
                print(f"  Progress: {i}/{len(matrix_names)} matrices loaded")

        print(f"✓ Preloaded {len(matrix_names)} matrices")

    def list_matrices(self) -> List[str]:
        """
        List all available matrix names.

        Returns
        -------
        List[str]
            List of matrix names (without .omx extension)
        """
        return sorted([f.stem for f in self.matrices_dir.glob("*.omx")])

    def get_metadata(self, matrix_name: str) -> dict:
        """
        Load metadata for a matrix.

        Parameters
        ----------
        matrix_name : str
            Name of the matrix

        Returns
        -------
        dict
            Matrix metadata
        """
        json_path = self.matrices_dir / f"{matrix_name}_meta.json"

        if not json_path.exists():
            return {}

        with open(json_path, "r") as f:
            return json.load(f)

    def clear_cache(self) -> None:
        """Clear all cached matrices from memory."""
        num_cached = len(self._cache)
        self._cache.clear()
        print(f"✓ Cleared {num_cached} matrices from cache")

    @property
    def num_cached(self) -> int:
        """Number of currently cached matrices."""
        return len(self._cache)

    def __contains__(self, matrix_name: str) -> bool:
        """Check if a matrix exists."""
        return (self.matrices_dir / f"{matrix_name}.omx").exists()

    def __len__(self) -> int:
        """Number of available matrices."""
        return len(self.list_matrices())

    def __repr__(self) -> str:
        return (
            f"MatrixLoader(matrices_dir='{self.matrices_dir}', "
            f"available={len(self)}, cached={self.num_cached})"
        )
