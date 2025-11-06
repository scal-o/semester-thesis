"""
Mini script to test the convergence of the static traffic assignment using
the best-knwon flow solution from the tntp files.
"""

from pathlib import Path
import geopandas as gpd
import click


@click.command("test")
@click.argument("network")
@click.option(
    "--path",
    default="data",
    show_default=True,
    help="The base path to the scenarios directory.",
)
def test_sta(
    network: str,
    path: str,
):
    """
    Checks results of the AequilibraE assignment for the given NETWORK against
    the best-known flow results from TNTP files.
    """
    print("--- Testing static traffic assignment ---")
    print(f"Network: {network}")
    print(f"Path: {path}")

    # setting up paths
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    scenarios_path = path / network
    output_path = scenarios_path / "scenarios_sta_results"

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

    if not output_path.is_dir():
        raise FileNotFoundError(
            f"Output path {output_path} does not exists.\nCould not find simulation results."
        )

    base_scenario = scenarios_path / "scenarios_geojson" / "scenario_00000"
    print(f"Testing assignment for {base_scenario}...")

    # read flows for best-knwon case and assignment results
    best_known_flows = gpd.read_file(base_scenario / "flows.geojson")
    sta_own_flows = gpd.read_parquet(output_path / "scenario_00000" / "scenario_00000.parquet")

    # check that both datasets have the same length (i.e. contain the same number of links)
    if len(best_known_flows) != len(sta_own_flows):
        raise ValueError("\nBest-known flows and own flows dataframes have different lengths.")

    # sort datasets and cast important columns to common dtypes
    best_known_flows.sort_values(
        by=["a_node", "b_node"], ascending=True, inplace=True, ignore_index=True
    )
    sta_own_flows.sort_values(
        by=["a_node", "b_node"], ascending=True, inplace=True, ignore_index=True
    )

    best_known_flows = best_known_flows.astype({"a_node": "int32", "b_node": "int32"})
    sta_own_flows = sta_own_flows.astype({"a_node": "int32", "b_node": "int32"})

    # check that both datasets have the same links
    if not best_known_flows[["a_node", "b_node"]].equals(sta_own_flows[["a_node", "b_node"]]):
        raise ValueError(
            "Best-known flows and own flows dataframes do not contain values for the same links."
        )

    # extract flows
    best_known_flows = best_known_flows["flow"].astype("float32")
    sta_own_flows = sta_own_flows["flow"].astype("float32")

    # define tolerance for the results (defaults to 5%)
    tol = best_known_flows.mean() * 0.05
    avg_diff = abs(best_known_flows - sta_own_flows).mean()

    print("--- Assignment Test Complete ---")
    print(f"Average difference between best-known flows and assignment results: {avg_diff:.3f}")
    print(f"Tolerance margin (5%): {tol:.3f}")

    if avg_diff < tol:
        print("Static assignment converged to best-known resuts.")
    elif avg_diff > tol:
        print("Static assignment did not converge to best-known results.")
