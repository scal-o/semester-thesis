"""
AssignmentBuilder: Run traffic assignments on scenario/matrix combinations.
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import json
import time
import pandas as pd
from aequilibrae.paths import Graph
from aequilibrae.paths.traffic_assignment import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass

from loaders import GraphLoader, ODMatrixLoader


@dataclass
class AssignmentConfig:
    """
    Configuration for traffic assignment parameters.

    Attributes
    ----------
    vdf : str
        Volume-delay function (default: "BPR")
    vdf_parameters : Dict[str, str]
        VDF parameter mapping (default: {"alpha": "b", "beta": "power"})
    algorithm : str
        Assignment algorithm (default: "msa")
    max_iter : int
        Maximum iterations (default: 100)
    rgap_target : float
        Relative gap convergence target (default: 1e-6)
    capacity_field : str
        Network capacity field (default: "capacity")
    time_field : str
        Network time field (default: "free_flow_time")
    traffic_class_name : str
        Name of traffic class (default: "car")
    """

    vdf: str = "BPR"
    vdf_parameters: Dict[str, str] = field(default_factory=lambda: {"alpha": "b", "beta": "power"})
    algorithm: str = "msa"
    max_iter: int = 100
    rgap_target: float = 1e-6
    capacity_field: str = "capacity"
    time_field: str = "free_flow_time"
    traffic_class_name: str = "car"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssignmentConfig":
        """Create from dictionary."""
        return cls(**data)

    def update(self, **kwargs) -> "AssignmentConfig":
        """Create a new config with updated parameters."""
        data = self.to_dict()
        data.update(kwargs)
        return AssignmentConfig.from_dict(data)


@dataclass
class AssignmentResult:
    """
    Result from a traffic assignment run.

    Attributes
    ----------
    scenario_name : str
        Network scenario used
    matrix_name : str
        OD matrix used
    result_name : str
        Unique identifier for this result
    elapsed_time : float
        Time taken in seconds
    config : Dict[str, Any]
        Assignment configuration used
    """

    scenario_name: str
    matrix_name: str
    result_name: str
    elapsed_time: float
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AssignmentBuilder:
    """
    Builder for running traffic assignments using GraphLoader and ODMatrixLoader.

    Combines network scenarios and OD matrices to run traffic assignments
    and store results with comprehensive metadata.
    """

    def __init__(
        self,
        graph_loader: GraphLoader,
        matrix_loader: ODMatrixLoader,
        results_dir: Union[str, Path] = "assignment_results",
        default_config: Optional[AssignmentConfig] = None,
    ):
        """
        Initialize AssignmentBuilder.

        Parameters
        ----------
        graph_loader : GraphLoader
            GraphLoader instance for loading network scenarios
        matrix_loader : ODMatrixLoader
            ODMatrixLoader instance for loading OD matrices
        results_dir : Union[str, Path], optional
            Directory to store assignment results (default: "assignment_results")
        default_config : AssignmentConfig, optional
            Default assignment configuration. If None, uses AssignmentConfig defaults.
        """
        self.graphs = graph_loader
        self.matrices = matrix_loader
        self.results_dir = Path(results_dir)

        # Create results subdirectories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.results_dir / "metadata"
        self.flows_dir = self.results_dir / "flows"
        self.metadata_dir.mkdir(exist_ok=True)
        self.flows_dir.mkdir(exist_ok=True)

        # Set default configuration
        self.default_config = default_config if default_config is not None else AssignmentConfig()

    def run_assignment(
        self,
        scenario_name: str,
        matrix_name: str,
        output_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> AssignmentResult:
        """
        Run a single traffic assignment.

        Parameters
        ----------
        scenario_name : str
            Name of network scenario (from GraphLoader)
        matrix_name : str
            Name of OD matrix (from ODMatrixLoader)
        output_name : str, optional
            Custom name for results. If None, uses "{scenario_name}_{matrix_name}"
        config_overrides : Dict[str, Any], optional
            Override specific default config parameters for this assignment

        Returns
        -------
        AssignmentResult
            Result summary with convergence info, timing, and file paths

        Examples
        --------
        >>> builder = AssignmentBuilder(graph_loader, matrix_loader)
        >>> result = builder.run_assignment("capacity_50_001", "peak_hour_001")

        >>> # Override config for specific run
        >>> result = builder.run_assignment(
        ...     "capacity_50_001",
        ...     "peak_hour_001",
        ...     config_overrides={"max_iter": 200, "rgap_target": 1e-8}
        ... )
        """
        # Create result name
        if output_name is None:
            output_name = f"{scenario_name}_{matrix_name}"

        # Merge config with overrides
        config = self.default_config
        if config_overrides:
            config = config.update(**config_overrides)

        print(f"Running assignment: {output_name}")
        print(f"  Scenario: {scenario_name}")
        print(f"  Matrix: {matrix_name}")

        # Get centroids from matrix
        centroids = self.matrices.get_centroids([matrix_name])[0]
        print(f"  Centroids: {len(centroids)} active zones")

        # Load graph with correct centroids
        graph = self.graphs.load(scenario_name, centroids)
        print(f"  Graph: {len(graph.network)} links")

        # Load matrix
        matrix = self.matrices.load(matrix_name)
        print(f"  Matrix: {matrix.matrix_view.sum():.0f} total demand")

        # Create traffic class
        traffic_class = TrafficClass(name=config.traffic_class_name, graph=graph, matrix=matrix)

        # Setup traffic assignment
        assignment = TrafficAssignment()
        assignment.set_classes([traffic_class])
        assignment.set_vdf(config.vdf)
        assignment.set_vdf_parameters(config.vdf_parameters)
        assignment.set_capacity_field(config.capacity_field)
        assignment.set_time_field(config.time_field)
        assignment.set_algorithm(config.algorithm)
        assignment.max_iter = config.max_iter
        assignment.rgap_target = config.rgap_target

        # Execute assignment
        print("Starting assignment...")
        start_time = time.perf_counter()
        assignment.execute()
        elapsed_time = time.perf_counter() - start_time

        # Save results to file
        self._save_results(output_name, assignment, graph, scenario_name, matrix_name, config)

        # Create result object
        result = AssignmentResult(
            scenario_name=scenario_name,
            matrix_name=matrix_name,
            result_name=output_name,
            elapsed_time=elapsed_time,
            config=config.to_dict(),
        )

        # Save metadata to file
        self._save_metadata(result)

        print(f"    Saved results to {self.results_dir / output_name}")

        return result

    def batch_run_assignments(
        self,
        scenario_names: Union[List[str], str] = "all",
        matrix_names: Union[List[str], str] = "all",
        config_overrides: Optional[Dict[str, Any]] = None,
        preload: bool = True,
    ) -> pd.DataFrame:
        """
        Run assignments for all combinations of scenarios and matrices.

        Parameters
        ----------
        scenario_names : Union[List[str], str], optional
            List of scenario names or "all" (default: "all")
        matrix_names : Union[List[str], str], optional
            List of matrix names or "all" (default: "all")
        config_overrides : Dict[str, Any], optional
            Override default config parameters for all assignments
        preload : bool, optional
            Whether to preload graphs and matrices into cache (default: True)

        Returns
        -------
        pd.DataFrame
            Summary of all assignments with convergence info

        Examples
        --------
        >>> # Run all combinations
        >>> results = builder.batch_run_assignments()

        >>> # Run specific combinations
        >>> results = builder.batch_run_assignments(
        ...     scenario_names=["capacity_50_001", "capacity_75_001"],
        ...     matrix_names=["peak_hour_001", "peak_hour_002"]
        ... )

        >>> # Override config for batch
        >>> results = builder.batch_run_assignments(
        ...     scenario_names="all",
        ...     matrix_names="all",
        ...     config_overrides={"max_iter": 150, "rgap_target": 1e-7}
        ... )
        """
        # Handle "all" keyword
        if scenario_names == "all":
            scenario_names = self.graphs.list_scenarios()
        if matrix_names == "all":
            matrix_names = self.matrices.list_matrices()

        if len(scenario_names) != len(matrix_names):
            raise ValueError(
                f"There should be an equal number of scenarios and matrices."
                f"Selected scenarios: {len(scenario_names)}"
                f"Selected matrices: {len(matrix_names)}"
            )

        # Optionally preload for efficiency
        if preload:
            print("Preloading data...")
            centroids = self.matrices.get_centroids(matrix_names)
            self.graphs.preload(scenario_names, centroids)
            self.matrices.preload(matrix_names)
            print()

        # Run all combinations
        results = []
        count = 0
        total = len(scenario_names)

        for scenario_name, matrix_name in zip(scenario_names, matrix_names):
            count += 1
            print(f"\n[{count}/{total}] - Running assignment: {scenario_name} × {matrix_name}")

            try:
                result = self.run_assignment(
                    scenario_name, matrix_name, config_overrides=config_overrides
                )
                results.append(result.to_dict())
            except Exception as e:
                print(f"    Failed: {e}")
                # Log failure but continue
                results.append(
                    {
                        "scenario_name": scenario_name,
                        "matrix_name": matrix_name,
                        "result_name": f"{scenario_name}_{matrix_name}",
                        "error": str(e),
                    }
                )

        print(f"\n{'#' * 70}")
        print(f"BATCH COMPLETE: {count} assignments finished")
        print(f"{'#' * 70}\n")

        results_df = pd.DataFrame(results)

        # Save batch summary
        summary_path = self.results_dir / "batch_summary.csv"
        results_df.to_csv(summary_path, index=False)
        print(f"Batch summary saved to {summary_path}\n")

        return results_df

    def _save_results(
        self,
        result_name: str,
        assignment: TrafficAssignment,
        graph: Graph,
        scenario_name: str,
        matrix_name: str,
        config: AssignmentConfig,
    ) -> None:
        """Save assignment results to disk."""

        # Get link flows and network data
        results = assignment.results()
        network = graph.network.copy()
        network.index = np.arange(1, len(network) + 1)

        # Add flows to network
        network["flow"] = results["PCE_AB"]
        network["volume_capacity_ratio"] = results["VOC_AB"]
        network["congested_time"] = results["Congested_Time_AB"]

        # Save flows as CSV
        flows_path = self.flows_dir / f"{result_name}.csv"
        network.to_csv(flows_path, index=False)

    def _save_metadata(self, result: AssignmentResult) -> None:
        """Save assignment metadata."""
        metadata_path = self.metadata_dir / f"{result.result_name}.json"

        with open(metadata_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def list_results(self) -> pd.DataFrame:
        """
        List all completed assignment results.

        Returns
        -------
        pd.DataFrame
            DataFrame with result summaries
        """
        metadata_files = sorted(self.metadata_dir.glob("*.json"))

        if not metadata_files:
            return pd.DataFrame()

        results = []
        for filepath in metadata_files:
            with open(filepath, "r") as f:
                results.append(json.load(f))

        return pd.DataFrame(results)

    def get_result(self, result_name: str) -> AssignmentResult:
        """
        Load a specific assignment result.

        Parameters
        ----------
        result_name : str
            Name of the result

        Returns
        -------
        AssignmentResult
            Result object with metadata
        """
        metadata_path = self.metadata_dir / f"{result_name}.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Result '{result_name}' not found. "
                f"Available results: {self.list_results()['result_name'].tolist()}"
            )

        with open(metadata_path, "r") as f:
            data = json.load(f)

        return AssignmentResult(**data)

    def get_link_flows(self, result_name: str) -> pd.DataFrame:
        """
        Get link flows from assignment result.

        Parameters
        ----------
        result_name : str
            Name of the result

        Returns
        -------
        pd.DataFrame
            Network with flow results
        """
        flows_path = self.flows_dir / f"{result_name}.csv"

        if not flows_path.exists():
            raise FileNotFoundError(f"Flows for '{result_name}' not found")

        return pd.read_csv(flows_path)

    def __repr__(self) -> str:
        num_results = len(list(self.metadata_dir.glob("*.json")))
        return (
            f"AssignmentBuilder(\n"
            f"  scenarios={len(self.graphs)}, matrices={len(self.matrices)}\n"
            f"  results={num_results}, results_dir='{self.results_dir}'\n"
            f")"
        )
