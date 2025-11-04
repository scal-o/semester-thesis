"""
Script for GeoPackage -> OpenMatrix (AequilibraE-compatible format) conversion.

This script reads geojson scenario demand files and converts them into OpenMatrix (.omx) format.
"""

from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import openmatrix as omx
import pandas as pd
from tqdm import tqdm


def convert_od(scenario_dir: Path) -> bool:
    """
    Converts a single scenario's OD table from a .gpkg file into an .omx file.
    Worker function meant to be used with multiprocessor for parallelization.

    Args:
        scenario_gpkg_path: Path to the input scenario_XXXX.gpkg.
        output_dir: The directory to save the new .omx file.

    Returns:
        True on success, False on failure.
    """
    output_file = scenario_dir / "od.omx"

    try:
        # 1. Load data from the single .gpkg file
        od_df = gpd.read_file(scenario_dir / "od.geojson")
        od_df = od_df[["origin", "destination", "demand"]]

        # 2. Get the sorted zone ids
        zone_ids = pd.concat([od_df["origin"], od_df["destination"]])
        zone_ids = sorted(zone_ids.unique())

        # 3. Pivot the long-format OD data into a wide matrix
        od_pivot = od_df.pivot_table(
            index="origin", columns="destination", values="demand", fill_value=0.0
        )

        # 4. Reindex the matrix to ensure N x N shape
        matrix_full = od_pivot.reindex(index=zone_ids, columns=zone_ids, fill_value=0.0)

        matrix_data = matrix_full.to_numpy(dtype=np.float32)

        # 5. --- Write to single .omx file ---
        # 'w' mode (write) will overwrite the file if it exists
        # Save OMX file
        with omx.open_file(str(output_file), "w") as omx_file:
            omx_file["matrix"] = matrix_data
            omx_file.create_mapping("taz", zone_ids)

        return True

    except Exception as e:
        print(f"Error processing {scenario_dir.name}: {e}")
        return False


@click.command()
@click.argument("network")
@click.option(
    "--path",
    default="data",
    show_default=True,
    help="The base path to the scenarios directory.",
)
@click.option(
    "-p",
    "--processes",
    type=int,
    default=None,
    help="Number of parallel processes to use. Defaults to number of CPU cores.",
)
def convert_to_omx(
    network: str,
    path: str = None,
    processes: int = None,
):
    """
    Converts each .gpkg scenario OD table for NETWORK into its own .omx file.
    """
    print("--- OpenMatrix conversion ---")
    print(f"Network: {network}")
    print(f"Path: {path}")

    # setting up paths
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    scenarios_path = path / network / "scenarios_geojson"

    if not scenarios_path.is_dir():
        raise ValueError(
            f"Scenarios path {scenarios_path} does not exist."
            "Make sure to run the script from the data directory or provide a base path."
        )

    # find all scenario files to convert
    scenario_files = list(scenarios_path.glob("scenario_*"))
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found in {scenarios_path}")

    print(f"Found {len(scenario_files)} scenario files to process.")
    print(f"Starting conversion of {len(scenario_files)} files to .omx format...")

    # create process pool
    if processes is None:
        processes = cpu_count()

    with Pool(processes=processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(convert_od, scenario_files),
                total=len(scenario_files),
                desc="Processing & Writing .omx",
            )
        )

    print("--- OpenMatrix File Creation Complete ---")
    success_count = sum(1 for res in results if res)
    print(f"Successfully wrote {success_count} / {len(scenario_files)} files.")
