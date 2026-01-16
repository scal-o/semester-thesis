"""Batch experiment runner for sequential training runs."""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from ml_static.config import load_config
from ml_static.run import run_training
from ml_static.utils import get_project_root


class ConfigManager:
    """Manages temporary config modifications for batch runs."""

    def __init__(self, config_path: Path | None = None):
        """Initialize config manager.

        Args:
            config_path: Path to the config file to manage. Defaults to conf_run.yaml.
        """
        if config_path is None:
            self.config_path = get_project_root() / "configs" / "conf_run.yaml"
        else:
            self.config_path = Path(config_path)

        self.backup_path = self.config_path.with_stem(self.config_path.stem + "_backup")
        self.original_content: dict = None

    def backup(self) -> None:
        """Create a backup of the current config."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.original_content = yaml.safe_load(f)

    def restore(self) -> None:
        """Restore config from backup."""
        if self.original_content is not None:
            with open(self.config_path, "w") as f:
                yaml.dump(self.original_content, f, default_flow_style=False)

    def update(self, updates: dict) -> None:
        """Update the config with new values.

        Args:
            updates: Dictionary with updates. Supports:
                - Simple keys: {"optimizer.learning_rate": 0.001}
                - List replacement: {"training.target.pre": [{"builder": "target_vcr"}]}
                - Full nested dicts: {"training.target": {"pre": [...], "post": [...]}}
        """
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Handle nested updates (e.g., "training.epochs" -> config["training"]["epochs"])
        for key, value in updates.items():
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


def run_batch_experiments(
    experiment_configs: list[dict], config_manager: ConfigManager | None = None
) -> dict:
    """Run multiple experiments sequentially by modifying conf_run.yaml.

    Args:
        experiment_configs: List of dicts, each containing:
            - "name": str - name of the experiment
            - "updates": dict - config updates to apply (nested keys like "training.epochs")
            - "description": str (optional) - description of the experiment

        config_manager: ConfigManager instance to use. If None, creates one.

    Returns:
        Dictionary with results for each experiment.
    """
    if config_manager is None:
        config_manager = ConfigManager()

    # Backup original config
    config_manager.backup()

    results = {}

    try:
        for exp in experiment_configs:
            exp_name = exp["name"]
            updates = exp.get("updates", {})
            description = exp.get("description", "")

            print(f"\n{'=' * 70}")
            print(f"Experiment: {exp_name}")
            if description:
                print(f"Description: {description}")
            print(f"{'=' * 70}")

            try:
                # Reset to clean state before applying updates for this experiment
                config_manager.restore()
                print("Config reset to baseline")

                # Update conf_run.yaml with experiment-specific values
                config_manager.update(updates)
                print(f"Config updated with: {updates}")

                # Load and run with updated config
                config = load_config(config_manager.config_path)
                model, dataset_split = run_training(config, check_run=False)

                results[exp_name] = {"status": "SUCCESS", "error": None}
                print(f"✓ {exp_name} completed successfully")

            except Exception as e:
                results[exp_name] = {"status": "FAILED", "error": str(e)}
                print(f"✗ {exp_name} failed: {e}")
                raise  # Re-raise to stop batch on error

    finally:
        # Always restore original config
        print(f"\n{'=' * 70}")
        print("Restoring original conf_run.yaml...")
        config_manager.restore()
        print("✓ Config restored")

    return results


def load_experiment_suite(suite_path: Path) -> list[dict]:
    """Load experiment suite from YAML file.

    Args:
        suite_path: Path to experiment suite YAML file.

    Returns:
        List of experiment configuration dicts.
    """
    with open(suite_path) as f:
        suite = yaml.safe_load(f)

    return suite.get("experiments", [])


@click.command("batch-train")
@click.option(
    "-s",
    "--suite",
    type=click.Path(exists=True),
    required=True,
    help="Path to experiment suite YAML file.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file to modify (defaults to conf_run.yaml).",
)
def batch_train(suite: str, config: str) -> None:
    """Run multiple experiments from a suite file by modifying the config in-place.

    The suite file should be a YAML with an 'experiments' list:

    experiments:
      - name: "experiment_1"
        description: "Optional description"
        updates:
          training.epochs: 50
          optimizer.learning_rate: 0.001

      - name: "experiment_2"
        description: "Another experiment"
        updates:
          dataset.name: "anaheim"
          training.batch_size: 16
    """
    suite_path = Path(suite)
    config_path = Path(config) if config else None

    print(f"Loading experiment suite from {suite_path}...")
    experiments = load_experiment_suite(suite_path)

    if not experiments:
        print("No experiments found in suite file.")
        return

    print(f"Found {len(experiments)} experiment(s)")

    config_manager = ConfigManager(config_path)

    try:
        results = run_batch_experiments(experiments, config_manager)

        # Print summary
        print(f"\n{'=' * 70}")
        print("BATCH RUN SUMMARY:")
        print(f"{'=' * 70}")
        for exp_name, result in results.items():
            status = result["status"]
            if status == "SUCCESS":
                print(f"  ✓ {exp_name}: {status}")
            else:
                print(f"  ✗ {exp_name}: {status}")
                if result["error"]:
                    print(f"      Error: {result['error']}")

    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user.")
        config_manager.restore()
        print("✓ Config restored")
    except Exception as e:
        print(f"\n\nBatch run failed with error: {e}")
        config_manager.restore()
        print("✓ Config restored")
        raise


if __name__ == "__main__":
    batch_train()
