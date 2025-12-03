"""
Configuration management for ml-static.

This module provides a clean separation between:
1. Configuration schema (dataclasses) - pure data with validation
2. Configuration loading - YAML parsing and schema construction
3. Factory methods remain on domain classes as thin wrappers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# =============================================================================
# Configuration Schema (Pure Data Structures)
# =============================================================================


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer."""

    type: Literal["adam", "sgd"] = "adam"
    learning_rate: float = 0.001
    momentum: float = 0.9  # only used for SGD

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got {self.learning_rate}")


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: Literal["reduce_on_plateau"] = "reduce_on_plateau"
    factor: float = 0.1
    patience: int = 10


@dataclass(frozen=True)
class LossConfig:
    """Configuration for loss function."""

    type: Literal["mse", "l1", "huber"] = "mse"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset."""

    name: str = "sioux_falls"
    path: Path = field(default_factory=lambda: Path("data"))
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if abs(sum(self.split) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(self.split)}")

    @property
    def full_path(self) -> Path:
        """Get full path to dataset directory."""
        return self.path / self.name


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 100
    batch_size: int = 32
    seed: int | None = None  # None means random seed


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model architecture."""

    name: str = "GNN"
    input_channels: int | None = None  # None means infer from data
    hidden_channels: int = 32
    output_channels: int = 16


@dataclass(frozen=True)
class TargetTransformConfig:
    """Configuration for target (label) transform."""

    edge_type: tuple[str, ...] = ("nodes", "real", "nodes")
    label: str = "edge_vcr"
    transform: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def target(self) -> tuple[tuple[str, ...], str]:
        """Get target as (edge_type, label) tuple."""
        return (self.edge_type, self.label)


@dataclass(frozen=True)
class BuilderTransformConfig:
    """Configuration for builder transform in post-pipeline."""

    builder: str


@dataclass(frozen=True)
class ScalerTransformConfig:
    """Configuration for scaler transform in post-pipeline."""

    type_spec: tuple[str, ...]  # Can be node_type or edge_type
    feature: str
    transform: str  # Required, no default
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def target(self) -> tuple[tuple[str, ...], str]:
        """Get target as (type_spec, feature) tuple."""
        return (self.type_spec, self.feature)


@dataclass(frozen=True)
class TransformsConfig:
    """Configuration for all transforms (target, pre, and post)."""

    target: TargetTransformConfig = field(default_factory=TargetTransformConfig)
    pre: tuple[BuilderTransformConfig, ...] = field(default_factory=tuple)
    post: tuple[BuilderTransformConfig | ScalerTransformConfig, ...] = field(default_factory=tuple)


@dataclass
class Config:
    """
    Root configuration object.

    This is the main entry point for all configuration. It holds
    all sub-configurations as typed, validated dataclasses.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)

    # keep raw config for MLflow logging
    _raw: dict = field(default_factory=dict, repr=False)

    @property
    def raw_config(self) -> dict:
        """Get raw configuration dictionary for logging."""
        return self._raw


# =============================================================================
# Configuration Loader
# =============================================================================


class ConfigLoader:
    """
    Loads configuration from YAML files into Config schema.

    Handles the complexity of:
    - Loading main run config
    - Loading model-specific config with transforms
    - Merging default transforms with overrides
    """

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        model_config_path: Path | None = None,
    ) -> Config:
        """
        Load configuration from YAML file(s).

        Args:
            config_path: Path to the main run configuration file.
            model_config_path: Optional path to model configuration.
                If None, auto-discovers based on model name.

        Returns:
            Fully constructed Config object.

        Raises:
            FileNotFoundError: If required config files don't exist.
            yaml.YAMLError: If YAML is malformed.
            ValueError: If configuration values are invalid.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # parse main config sections
        model = cls._parse_model_config(raw_config.get("model", {}))
        training = cls._parse_training_config(raw_config.get("training", {}))
        optimizer = cls._parse_optimizer_config(raw_config.get("optimizer", {}))
        scheduler = cls._parse_scheduler_config(raw_config.get("scheduler", {}))
        loss = cls._parse_loss_config(raw_config.get("loss", {}))
        dataset = cls._parse_dataset_config(raw_config.get("dataset", {}))

        # parse target transform from training section
        target_transform = cls._parse_target_transform(raw_config.get("training", {}))

        # load model-specific config for pre and post transforms
        pre_transforms: tuple[BuilderTransformConfig, ...] = tuple()
        post_transforms: tuple[BuilderTransformConfig | ScalerTransformConfig, ...] = tuple()
        if model_config_path is None:
            model_config_path = config_path.parent / f"conf_model_{model.name}.yaml"

        if model_config_path.exists():
            pre_transforms, post_transforms = cls._load_model_transforms(model_config_path)

        transforms = TransformsConfig(
            target=target_transform,
            pre=pre_transforms,
            post=post_transforms,
        )

        return Config(
            model=model,
            training=training,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            dataset=dataset,
            transforms=transforms,
            _raw=raw_config,
        )

    @classmethod
    def _parse_model_config(cls, data: dict) -> ModelConfig:
        """Parse model configuration section."""
        return ModelConfig(
            name=data.get("name", "GNN"),
            input_channels=data.get("input_channels"),
            hidden_channels=data.get("hidden_channels", 32),
            output_channels=data.get("output_channels", 16),
        )

    @classmethod
    def _parse_training_config(cls, data: dict) -> TrainingConfig:
        """Parse training configuration section."""
        return TrainingConfig(
            epochs=data.get("epochs", 100),
            batch_size=data.get("batch_size", 32),
            seed=data.get("seed"),
        )

    @classmethod
    def _parse_optimizer_config(cls, data: dict) -> OptimizerConfig:
        """Parse optimizer configuration section."""
        return OptimizerConfig(
            type=data.get("type", "adam").lower(),
            learning_rate=data.get("learning_rate", 0.001),
            momentum=data.get("momentum", 0.9),
        )

    @classmethod
    def _parse_scheduler_config(cls, data: dict) -> SchedulerConfig:
        """Parse scheduler configuration section."""
        return SchedulerConfig(
            type=data.get("type", "reduce_on_plateau").lower(),
            factor=data.get("factor", 0.1),
            patience=data.get("patience", 10),
        )

    @classmethod
    def _parse_loss_config(cls, data: dict) -> LossConfig:
        """Parse loss configuration section."""
        loss_type = data.get("type", "mse")
        kwargs = {k: v for k, v in data.items() if k != "type"}
        return LossConfig(type=loss_type, kwargs=kwargs)

    @classmethod
    def _parse_dataset_config(cls, data: dict) -> DatasetConfig:
        """Parse dataset configuration section."""
        split_dict = data.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
        split = tuple(split_dict.values()) if isinstance(split_dict, dict) else tuple(split_dict)

        return DatasetConfig(
            name=data.get("name", "sioux_falls"),
            path=Path(data.get("path", "data")),
            split=split,
        )

    @classmethod
    def _parse_target_transform(cls, training_data: dict) -> TargetTransformConfig:
        """Parse target transform from training section."""
        target = training_data.get("target", {})

        edge_type = target.get("type", ["nodes", "real", "nodes"])
        if isinstance(edge_type, list):
            edge_type = tuple(edge_type)

        return TargetTransformConfig(
            edge_type=edge_type,
            label=target.get("label", "edge_vcr"),
            transform=target.get("transform"),
        )

    @classmethod
    def _load_model_transforms(
        cls, model_config_path: Path
    ) -> tuple[
        tuple[BuilderTransformConfig, ...],
        tuple[BuilderTransformConfig | ScalerTransformConfig, ...],
    ]:
        """
        Load model-specific transforms.

        Returns:
            Tuple of (pre_transforms, post_transforms).
        """
        # load model-specific config
        with open(model_config_path) as f:
            model_data = yaml.safe_load(f) or {}

        model_transforms = model_data.get("transforms", {}) or {}
        model_pre = model_transforms.get("pre", []) or []
        model_post = model_transforms.get("post", []) or []

        # parse pre-transforms (list-based)
        pre_transform_configs: list[BuilderTransformConfig] = []

        if isinstance(model_pre, list):
            for item in model_pre:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Pre-transform item must be a dict, got {type(item).__name__}"
                    )

                if "builder" in item:
                    pre_transform_configs.append(BuilderTransformConfig(builder=item["builder"]))
                else:
                    raise NotImplementedError(
                        f"Pre-transforms only support 'builder' key. Got: {list(item.keys())}"
                    )

        pre_transforms = tuple(pre_transform_configs)

        # parse post-transforms (list-based)
        post_transforms: list[BuilderTransformConfig | ScalerTransformConfig] = []

        if isinstance(model_post, list):
            for item in model_post:
                if not isinstance(item, dict):
                    continue

                # Check if it's a builder or scaler
                if "builder" in item:
                    # Builder transform
                    post_transforms.append(BuilderTransformConfig(builder=item["builder"]))
                elif "scaler" in item:
                    # Scaler transform (nested config)
                    scaler_config = item["scaler"]

                    # Parse type_spec:
                    # - list of len > 1: edge type (can be a list in case of edge, or str for node)
                    # - list of len 1 | str: single node type (str)
                    type_spec_config = scaler_config.get("type")
                    if not type_spec_config:
                        raise ValueError("Scaler must specify 'type'")

                    if isinstance(type_spec_config, list):
                        type_spec = tuple(type_spec_config)
                        if len(type_spec) == 1:
                            type_spec = type_spec[0]
                    elif isinstance(type_spec_config, str):
                        type_spec = type_spec_config
                    else:
                        raise ValueError(
                            f"Type spec must be string or list, got {type(type_spec_config).__name__}"
                        )

                    # parse feature
                    feature = scaler_config.get("feature")
                    if not feature:
                        raise ValueError("Scaler must specify 'feature'")

                    # parse transform type
                    transform_type = scaler_config.get("transform")
                    if not transform_type:
                        raise ValueError(
                            "Scaler must specify 'transform' type (e.g., 'minmax', 'norm', 'log')"
                        )

                    # parse transform kwargs
                    kwargs = {
                        k: v
                        for k, v in scaler_config.items()
                        if k not in ("type", "feature", "transform")
                    }

                    # create scaler config
                    post_transforms.append(
                        ScalerTransformConfig(
                            type_spec=type_spec,
                            feature=feature,
                            transform=transform_type,
                            kwargs=kwargs,
                        )
                    )

        return pre_transforms, tuple(post_transforms)


# =============================================================================
# Convenience function for loading
# =============================================================================


def load_config(
    config_path: Path,
    model_config_path: Path | None = None,
) -> Config:
    """
    Load configuration from YAML file(s).

    This is the main entry point for loading configuration.

    Args:
        config_path: Path to the main run configuration file.
        model_config_path: Optional path to model configuration.

    Returns:
        Fully constructed Config object.

    Example:
        >>> config = load_config(Path("conf_run.yaml"))
        >>> config.training.epochs
        100
        >>> config.optimizer.learning_rate
        0.001
    """
    return ConfigLoader.from_yaml(config_path, model_config_path)
