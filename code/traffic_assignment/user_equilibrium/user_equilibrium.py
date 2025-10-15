"""
Full script to run user equilibrium assignment on a selected number of scenarios
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from builders import NetworkBuilder, ODMatrixBuilder
from loaders import GraphLoader, ODMatrixLoader
from assignment import AssignmentConfig, AssignmentBuilder

# config
NETWORK_NAME = "sioux_falls"
NETWORK_DIR = "networks/"
SCENARIOS_DIR = "scenarios"
MATRICES_DIR = "matrices"

NETWORK_FILE_NAME = "net.tntp"
OD_FILE_NAME = "trips.tntp"

cwd = Path(os.getcwd())
root_dir = cwd / NETWORK_DIR / NETWORK_NAME

NETFILE = root_dir / NETWORK_FILE_NAME
ODFILE = root_dir / OD_FILE_NAME

# create scenarios and matrices
n = 5  # n of scenarios/matrices


# function to create the network scenarios
def net_fun(x: pd.Series) -> pd.Series:
    return x * np.random.uniform(0.8, 1, size=len(x))


# function to create the demand scenarios
def od_fun(x: np.ndarray) -> np.ndarray:
    return x * np.random.uniform(0.5, 1.5, size=x.shape)


network_builder = NetworkBuilder(NETFILE, scenarios_dir=root_dir / SCENARIOS_DIR)
network_builder.clear_scenarios(True)
network_builder.build_scenarios("capacity", net_fun, n)
print(f"Created {n} network scenarios.")

od_builder = ODMatrixBuilder(ODFILE, matrices_dir=root_dir / MATRICES_DIR)
od_builder.clear_matrices(True)
od_builder.build_matrices(od_fun, n)
print(f"Created {n} demand scenarios.")


# create graph and matrix loaders
scenario_loader = GraphLoader(root_dir / SCENARIOS_DIR)
matrix_loader = ODMatrixLoader(root_dir / MATRICES_DIR)

# create assignment builder / runner with explicit config
config = AssignmentConfig(
    vdf="BPR",
    vdf_parameters={"alpha": "b", "beta": "power"},
    # bi-conjugated Franke-Wolfe
    algorithm="bfw",
    max_iter=100,
    rgap_target=1e-5,
    # graph fields used by AequilibraE traffic assignment
    capacity_field="capacity",
    time_field="free_flow_time",
    # traffic class for which to perform the assignment
    traffic_class_name="car",
)

assignment = AssignmentBuilder(
    graph_loader=scenario_loader,
    matrix_loader=matrix_loader,
    results_dir=root_dir / "results",
    default_config=config,
)


# s = scenario_loader.list_scenarios()[0]
# m = matrix_loader.list_matrices()[0]
# assignment.run_assignment(s, m, "prova.json")


assignment.batch_run_assignments(scenario_names="all", matrix_names="all", preload=True)
