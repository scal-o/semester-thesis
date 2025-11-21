import random
from pathlib import Path
from typing import Callable

import torch
import yaml


class Config:
    """
    Configuration loader for training runs.

    Loads all training parameters from a YAML file and provides
    convenient access to model, training, optimizer, and loss configurations.
    """

    def __init__(self, config_path: Path) -> None:
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    def get_loss_function(self) -> torch.nn.Module:
        """
        Instantiate loss function based on configuration.

        Returns:
            Initialized loss function module.

        Raises:
            ValueError: If loss function type is not supported.
        """
        loss_config = self._config.get("loss", {})
        loss_type = loss_config.get("type", "mse").lower()

        if loss_type == "mse":
            return torch.nn.MSELoss()
        elif loss_type == "l1":
            return torch.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function type: {loss_type}")

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        """
        Instantiate optimizer based on configuration.

        Args:
            model_params: Model parameters to optimize.

        Returns:
            Initialized optimizer.

        Raises:
            ValueError: If optimizer type is not supported.
        """
        optimizer_config = self._config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adam").lower()
        learning_rate = optimizer_config.get("learning_rate", 0.001)

        if optimizer_type == "adam":
            return torch.optim.Adam(model_params, lr=learning_rate)
        elif optimizer_type == "sgd":
            momentum = optimizer_config.get("momentum", 0.9)
            return torch.optim.SGD(model_params, lr=learning_rate, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def get_target_extractor(self) -> Callable:
        """
        Get target extractor function based on configuration.

        Returns:
            Target extractor callable.

        Raises:
            ValueError: If target type is not supported.
        """
        target_type = self._config.get("training", {}).get("target", "edge_labels")

        # return basic get_labels function
        if target_type == "edge_labels":

            def get_labels(g):
                return g["nodes", "real", "nodes"].edge_labels

            return get_labels

        # return normalized get_labels function
        elif target_type == "normalized_edge_labels":

            def get_normalized(g):
                labels = g["nodes", "real", "nodes"].edge_labels
                return (labels - labels.mean()) / (labels.std() + 1e-8)

            return get_normalized

        else:
            raise ValueError(f"Unknown target type: {target_type}")

    @property
    def epochs(self) -> int:
        """Get number of training epochs."""
        return self._config.get("training", {}).get("epochs", 100)

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self._config.get("training", {}).get("batch_size", 1)

    @property
    def hidden_channels(self) -> int:
        """Get hidden channels for model."""
        return self._config.get("model", {}).get("hidden_channels", 32)

    @property
    def output_channels(self) -> int:
        """Get output channels for model."""
        return self._config.get("model", {}).get("output_channels", 16)

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._config.get("model", {}).get("name", "GNN")

    @property
    def dataset_path(self) -> str:
        """Get dataset name."""
        ddir = self._config.get("dataset", {}).get("path", "data")
        dname = self._config.get("dataset", {}).get("name", "sioux_falls")
        return Path(ddir) / dname

    @property
    def raw_config(self) -> dict:
        """Get raw configuration dictionary for logging."""
        return self._config

    @property
    def seed(self) -> int:
        """Get random seed."""
        return self._config.get("training", {}).get("seed", random.randint(0, 9999))
