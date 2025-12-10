from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch_geometric
import yaml
from tabulate import tabulate
from tqdm import tqdm

from ml_static import reporting as rep
from ml_static.utils import get_project_root

if TYPE_CHECKING:
    from torch_geometric.data import Dataset

    from ml_static.data import DatasetSplit, STADataset


# env vars to load
ENV_VARS = {
    # force mlflow to use uv to register all deps
    "MLFLOW_LOCK_MODEL_DEPENDENCIES": "true",
}

# load env vars
env = os.environ
for key, value in ENV_VARS.items():
    env[key] = str(value)


# create tracking class
class MLflowtracker:
    def __init__(self) -> None:
        """Initialize MLflow tracker with configuration from conf_mlflow.yaml."""
        # retrieve configs
        # run_conf contains mlflow configs (tracking uri and experiment name)
        # run_params contains model parameters (name, epochs, loss, optimizer)
        with open(get_project_root() / "configs" / "conf_mlflow.yaml") as f:
            config = yaml.safe_load(f)

        # set uri and experiment
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(config["experiment"])

        # create losses lists
        self.train_losses: list[float] = list()
        self.val_losses: list[float] = list()

        # create loss components tracking (flat structure)
        self.train_components: dict[str, list[float]] = {}
        self.val_components: dict[str, list[float]] = {}

    def log_params(self, params: dict) -> None:
        """Log multiple parameters to MLflow.

        Args:
            params: Dictionary of parameter names and values.
        """

        # flatten parameters dictionary
        params = pd.json_normalize(params).to_dict(orient="records")[0]
        mlflow.log_params(params)

    def log_seed(self, seed: int) -> None:
        """Log random seed used for the run.

        Args:
            seed: Random seed value.
        """
        mlflow.log_param("seed", seed)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_components: dict | None = None,
        val_components: dict | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Log metrics for a single epoch.

        Args:
            epoch: Epoch number.
            train_loss: Training loss value.
            val_loss: Validation loss value.
            train_components: Dictionary of training loss components.
            val_components: Dictionary of validation loss components.
            learning_rate: Current learning rate.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        # track and log train components
        if train_components:
            for key, value in train_components.items():
                if key not in self.train_components:
                    self.train_components[key] = []
                self.train_components[key].append(value)
                metrics[f"train_{key}"] = value

        # track and log val components
        if val_components:
            for key, value in val_components.items():
                if key not in self.val_components:
                    self.val_components[key] = []
                self.val_components[key].append(value)
                metrics[f"val_{key}"] = value

        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        mlflow.log_metrics(metrics, step=epoch)

    def log_configs(self) -> None:
        """
        Log all configuration files found in the project's 'configs' directory
        to a 'configs' artifact directory in MLflow.
        """
        configs_dir = get_project_root() / "configs"
        if not configs_dir.is_dir():
            print(f"Warning: Configuration directory not found at {configs_dir}")
            return

        # log all config files
        mlflow.log_artifacts(str(configs_dir.absolute()), artifact_path="configs")

    def log_test_loss(self, test_loss: float) -> None:
        """Log test loss metric.

        Args:
            test_loss: Test loss value.
        """
        mlflow.log_metric("test_loss", test_loss)

    def log_best_model_info(self, best_epoch: int, best_val_loss: float) -> None:
        """Log information about the best model.

        Args:
            best_epoch: Epoch number where best validation loss was achieved.
            best_val_loss: Best validation loss value.
        """
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)

    def log_dataset_info(self, dataset_split) -> None:
        """Log dataset split information.

        Args:
            dataset_split: DatasetSplit instance containing split indices.
        """
        # log split sizes
        mlflow.log_params(
            {
                "train_size": len(dataset_split.indices["train"]),
                "val_size": len(dataset_split.indices["val"]),
                "test_size": len(dataset_split.indices["test"]),
            }
        )

    def log_training_curves(self) -> None:
        """Generate and log training curves for total loss and components."""
        # plot total loss
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        ax.plot(self.train_losses, label="Training loss")
        ax.plot(self.val_losses, label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss Curves")
        ax.legend()

        mlflow.log_figure(fig, "training_stats/training_curves.png")
        plt.close(fig)

        # if no components, return early
        if not self.train_components and not self.val_components:
            return

        # extract unique component names (without prefix)
        unweighted_components = set()
        weighted_train_components = {}
        weighted_val_components = {}

        for key in self.train_components:
            if key.startswith("unweighted_"):
                unweighted_components.add(key.replace("unweighted_", ""))
            elif key.startswith("weighted_"):
                weighted_train_components[key] = self.train_components[key]

        for key in self.val_components:
            if key.startswith("unweighted_"):
                unweighted_components.add(key.replace("unweighted_", ""))
            elif key.startswith("weighted_"):
                weighted_val_components[key] = self.val_components[key]

        # plot unweighted components (one plot per component with train and val)
        for component_name in unweighted_components:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(111)

            train_key = f"unweighted_{component_name}"
            val_key = f"unweighted_{component_name}"

            if train_key in self.train_components:
                ax.plot(
                    self.train_components[train_key],
                    label=f"Training {component_name.replace('_', ' ')}",
                )
            if val_key in self.val_components:
                ax.plot(
                    self.val_components[val_key],
                    label=f"Validation {component_name.replace('_', ' ')}",
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Unweighted {component_name.replace('_', ' ').title()}")
            ax.legend()

            mlflow.log_figure(fig, f"training_stats/unweighted_{component_name}.png")
            plt.close(fig)

        # plot weighted components (all train in one plot, all val in another)
        if weighted_train_components:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(111)

            for key, values in weighted_train_components.items():
                label = key.replace("weighted_", "").replace("_", " ").title()
                ax.plot(values, label=label)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Weighted Training Loss Components")
            ax.legend()

            mlflow.log_figure(fig, "training_stats/weighted_train_components.png")
            plt.close(fig)

        if weighted_val_components:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(111)

            for key, values in weighted_val_components.items():
                label = key.replace("weighted_", "").replace("_", " ").title()
                ax.plot(values, label=label)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Weighted Validation Loss Components")
            ax.legend()

            mlflow.log_figure(fig, "training_stats/weighted_val_components.png")
            plt.close(fig)

    def log_model(self, model: torch.nn.Module, model_type: str, data) -> None:
        """Log model with related artifacts.

        Logs:
        - model state dict (via PyTorch)
        - source code (ml_static module)
        - model summary (via torch_geometric)

        Args:
            model: The trained model to log.
            model_type: Type/name of the model (e.g., 'HetGAT').
            data: Sample data for generating model summary.
        """

        code_dir = Path(__file__).parent
        model.cpu()
        data.to("cpu")

        # log source code and model artifacts
        mlflow.log_artifacts(str(code_dir.absolute()), artifact_path="code")

        with tempfile.TemporaryDirectory() as dirname:
            tempdir = Path(dirname)

            # save model checkpoint
            checkpoint = model.extract_checkpoint()
            torch.save(checkpoint, tempdir / "model.pt")

            # save model summary
            with open(tempdir / "model_summary.txt", "w") as f:
                model_summary = torch_geometric.nn.summary(model, data)
                f.write(model_summary)

            # log model-related artifacts
            mlflow.log_artifacts(str(tempdir.absolute()), artifact_path="model")

    def log_all_performance_reports(
        self,
        model: torch.nn.Module,
        datasets: dict[str, Dataset],
    ) -> pd.DataFrame:
        """
        Convenience method that logs performance reports for multiple dataset splits.

        This method logs performance reports for each split (train, validation, test)
        and returns a combined statistics DataFrame.

        Args:
            model: The trained GNN model.
            datasets: Dictionary mapping dataset names to Datasets.

        Returns:
            A DataFrame containing statistics for all dataset splits.
        """
        stats_dfs = []
        figs = {}

        for dataset_name, dataset in tqdm(datasets.items(), desc="Logging Performance Reports"):
            pred_df = rep.compute_predictions(model, dataset)
            pred_df = rep.compute_errors(pred_df)
            stats_df = rep.compute_statistics(pred_df)
            stats_df["Dataset"] = dataset_name
            stats_dfs.append(stats_df)

            fig = rep.plot_performance_diagnostics(pred_df, dataset_name)
            figs[dataset_name] = fig

        stats_dfs = pd.concat(stats_dfs).set_index("Dataset")

        # log statistics df
        with tempfile.TemporaryDirectory() as dirname:
            tempdir = Path(dirname)
            with open(tempdir / "training_stats.txt", "w") as f:
                stats = tabulate(stats_dfs, headers="keys", tablefmt="psql")
                f.write(stats)
            tmp = str((tempdir / "training_stats.txt").absolute())
            mlflow.log_artifact(tmp, "training_stats")
        mlflow.log_dict(stats_dfs.to_dict(orient="index"), "training_stats/training_stats.json")

        # log diagnostic figures
        for dataset_name, fig in figs.items():
            mlflow.log_figure(fig, f"training_stats/diagnostics_{dataset_name}.png")
            plt.close(fig)

        return stats

    def log_scenario_predictions(
        self,
        model: torch.nn.Module,
        dataset: STADataset,
        dataset_name: str,
        scenario_index: int,
    ) -> None:
        """
        High-level method that generates and logs scenario prediction plots.

        This method internally:
        1. Generates predictions using reporting.plot_predictions()
        2. Logs the plot to MLflow

        Args:
            model: The trained GNN model.
            dataset: The dataset containing the scenario.
            scenario_index: The index of the scenario to plot.
        """

        # Generate and plot predictions
        fig = rep.plot_predictions(model, dataset, dataset_name, scenario_index)

        # Log to MLflow
        scenario_name = dataset.scenario_names[scenario_index]
        mlflow.log_figure(
            fig, f"scenario_predictions/{dataset_name}_{scenario_name}_predictions.png"
        )
        plt.close(fig)

    def log_random_scenario_predictions(
        self,
        model: torch.nn.Module,
        datasets: dict[str, Dataset],
        num_scenarios: int = 5,
    ) -> None:
        """
        Log prediction plots for random scenarios from each dataset split.

        Selects random scenarios from train, validation, and test splits
        and generates prediction plots for each.

        Args:
            model: The trained GNN model.
            datasets: Dictionary mapping split names to datasets
                     (e.g., {"train": train_dataset, "val": val_dataset, "test": test_dataset}).
            num_scenarios: Number of random scenarios to plot per split. Default is 5.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        print(f"Logging scenario predictions ({num_scenarios} per split)...")

        for split_name, dataset in tqdm(datasets.items(), desc="Logging Scenario Predictions"):
            # Ensure we don't try to sample more scenarios than available
            num_to_sample = min(num_scenarios, len(dataset))

            # Sample random scenario indices
            scenario_indices = np.random.choice(len(dataset), size=num_to_sample, replace=False)

            for scenario_idx in scenario_indices:
                try:
                    self.log_scenario_predictions(model, dataset, split_name, scenario_idx)
                except Exception as e:
                    scenario_name = dataset.scenario_names[scenario_idx]
                    print(
                        f"Warning: Failed to log predictions for {split_name}/{scenario_name}: {e}"
                    )
