import os
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch_geometric
import yaml
from numpy.typing import NDArray
from tabulate import tabulate
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ml_static import reporting as rep
from ml_static.data import STADataset

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
        with open(Path(__file__).parent / "conf_mlflow.yaml") as f:
            config = yaml.safe_load(f)

        # set uri and experiment
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(config["experiment"])

        # create losses lists
        self.train_losses: list[float] = list()
        self.val_losses: list[float] = list()

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
        learning_rate: float | None = None,
    ) -> None:
        """Log metrics for a single epoch.

        Args:
            epoch: Epoch number.
            train_loss: Training loss value.
            val_loss: Validation loss value.
            learning_rate: Current learning rate.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        mlflow.log_metrics(metrics, step=epoch)

    def log_test_loss(self, test_loss: float) -> None:
        """Log test loss metric.

        Args:
            test_loss: Test loss value.
        """
        mlflow.log_metric("test_loss", test_loss)

    def log_split_indices(self, train_idx: NDArray, val_idx: NDArray, test_idx: NDArray) -> None:
        """Log train/val/test split indices as artifacts.

        Args:
            train_idx: Array of training set indices.
            val_idx: Array of validation set indices.
            test_idx: Array of test set indices.
        """

        with tempfile.TemporaryDirectory() as dirname:
            tempdir = Path(dirname)

            # save indices to temporary files
            np.savetxt(tempdir / "train_indices.txt", train_idx, fmt="%d")
            np.savetxt(tempdir / "val_indices.txt", val_idx, fmt="%d")
            np.savetxt(tempdir / "test_indices.txt", test_idx, fmt="%d")

            # log as artifacts
            mlflow.log_artifacts(tempdir, artifact_path="split_indices")

    def log_training_curves(self) -> None:
        """Generate and log training curves."""
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        # loss subplot
        ax.plot(self.train_losses, label="Training loss")
        ax.plot(self.val_losses, label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss curves")
        ax.legend()

        # save and log plot
        mlflow.log_figure(fig, "training_stats/training_curves.png")
        plt.close(fig)

    def log_model(self, model: torch.nn.Module, model_name: str, data) -> None:
        """Log model with related artifacts.

        Logs:
        - model weights (via PyTorch)
        - source code (ml_static module)
        - model summary (via torch_geometric)

        Args:
            model: The trained model to log.
            model_name: Name of the model.
            data: Sample data for generating model summary.
        """

        code_dir = Path(__file__).parent
        model.cpu()
        data.to("cpu")

        with tempfile.TemporaryDirectory() as dirname:
            tempdir = Path(dirname)

            # save source code and config
            # TODO: if passing custom config to the cli, copy that instead of the default
            shutil.copytree(code_dir, tempdir, dirs_exist_ok=True)

            # save model weights and initialization parameters
            checkpoint = {
                "state_dict": model.state_dict(),
                "init_params": model._init_params,
            }
            torch.save(checkpoint, tempdir / "model.pt")

            # save model summary
            with open(tempdir / "model_summary.txt", "w") as f:
                model_summary = torch_geometric.nn.summary(model, data)
                f.write(model_summary)

            # log artifacts
            mlflow.log_artifacts(tempdir, artifact_path="code")

    def log_all_performance_reports(
        self,
        model: torch.nn.Module,
        loaders: dict[str, DataLoader],
    ) -> pd.DataFrame:
        """
        Convenience method that logs performance reports for multiple dataset splits.

        This method logs performance reports for each split (train, validation, test)
        and returns a combined statistics DataFrame.

        Args:
            model: The trained GNN model.
            loaders: Dictionary mapping dataset names to DataLoaders.

        Returns:
            A DataFrame containing statistics for all dataset splits.
        """
        stats_dfs = []
        figs = {}

        for dataset_name, loader in tqdm(loaders.items(), desc="Logging Performance Reports"):
            pred_df = rep.compute_predictions(model, loader)
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
            mlflow.log_artifact(tempdir / "training_stats.txt", "training_stats")
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
        mlflow.log_figure(fig, f"scenario_predictions/{scenario_name}_predictions.png")
        plt.close(fig)
