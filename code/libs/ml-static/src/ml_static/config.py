import random
from pathlib import Path

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

    def get_loss_type(self) -> str:
        """
        Get loss type based on configuration.
        """
        return self._config.get("loss", {}).get("type", "mse")

    def get_loss_params(self) -> dict:
        """
        Get loss params (needed for custom loss) from configuration.
        """
        params = self._config.get("loss", {}).copy()
        params.pop("type", None)
        return params

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

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Instantiate learning rate scheduler based on configuration.

        Args:
            optimizer: The optimizer for which to schedule the learning rate.

        Returns:
            Initialized learning rate scheduler.

        Raises:
            ValueError: If scheduler type is not supported.
        """
        scheduler_config = self._config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "reduce_on_plateau").lower()

        if scheduler_type == "reduce_on_plateau":
            factor = scheduler_config.get("factor", 0.1)
            patience = scheduler_config.get("patience", 10)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def get_target(self) -> tuple[tuple, str]:
        """
        Get target (labels) based on configuration.

        Returns:
            Target name.
        """
        target = self._config.get("training", {}).get("target", {})

        target_type = target.get("type", ("nodes", "real", "nodes"))
        if isinstance(target_type, list):
            target_type = tuple(target_type)
        target_label = target.get("label", "edge_vcr")

        return target_type, target_label

    def get_transform(self) -> str:
        """
        Get data transform type based on configuration.

        Returns:
            Transform function type.
        """
        transform = self._config.get("dataset", {}).get("transform", None)

        return transform

    @property
    def epochs(self) -> int:
        """Get number of training epochs."""
        return self._config.get("training", {}).get("epochs", 100)

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self._config.get("training", {}).get("batch_size", 1)

    @property
    def input_channels(self) -> int:
        """
        Get input channels for model. This is a REQUIRED parameter.
        Raises an error if not specified in the configuration.
        """
        channels = self._config.get("model", {}).get("input_channels", None)
        if channels is None:
            raise ValueError("Input channels not specified in configuration")
        return channels

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
