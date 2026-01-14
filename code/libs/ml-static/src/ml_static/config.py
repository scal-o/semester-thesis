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
from typing import Any, Literal, Protocol

import yaml

# =============================================================================
# Model Architecture Protocol
# =============================================================================


class ModelArchitectureConfig(Protocol):
    """
    Protocol that all model architecture configs must satisfy.

    This defines the minimal interface without forcing inheritance.
    Each model can define its own dataclass structure in its module.
    """

    @classmethod
    def from_dict(cls, data: dict) -> ModelArchitectureConfig:
        """Parse config from dictionary.

        Args:
            data: Dictionary containing architecture configuration.

        Returns:
            Parsed architecture config instance.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        ...

    def validate(self) -> None:
        """Validate architecture-specific constraints."""
        ...


# =============================================================================
# Configuration Schema (Pure Data Structures)
# =============================================================================


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer."""

    type: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = 0.001

    # SGD parameters
    momentum: float = 0.9

    # Adam/AdamW parameters
    weight_decay: float = 0.0
    amsgrad: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got {self.learning_rate}")


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: Literal["reduce_on_plateau", "one_cycle", "cosine_annealing"] = "reduce_on_plateau"

    # ReduceLROnPlateau parameters
    factor: float = 0.5
    patience: int = 30

    # OneCycleLR parameters
    max_lr: float = 0.01
    epochs: int | None = None
    steps_per_epoch: int | None = None
    pct_start: float = 0.3
    anneal_strategy: Literal["cos", "linear"] = "cos"

    # CosineAnnealingWarmRestarts parameters
    T_0: int = 20
    T_mult: int = 2
    eta_min: float = 1e-6


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    enabled: bool = False
    patience: int = 50
    min_delta: float = 0.0
    mode: Literal["min", "max"] = "min"


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


@dataclass
class ModelConfigWithArchitecture:
    """Model configuration with typed architecture."""

    type: str = "HetGAT"

    # Architecture loaded dynamically based on type
    architecture: ModelArchitectureConfig | None = None


@dataclass(frozen=True)
class BuilderTransformConfig:
    """Configuration for builder transform in pipeline."""

    builder: str


@dataclass(frozen=True)
class ScalerTransformConfig:
    """
    Unified configuration for scaler transforms.

    Handles both target scalers and feature scalers:
    - Target scalers: specify target="y"
    - Feature scalers: specify type="nodes" (or edge type) and feature="demand"
    """

    transform: str  # Required: 'log', 'norm', 'minmax'

    # For target scalers (labels): specify target="y"
    target: str | None = None

    # For feature scalers: specify type and feature
    type: str | list[str] | None = None  # node type or edge type (list for edges)
    feature: str | None = None

    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that only one of the two configurations is specified."""
        # check that exactly one configuration is used
        has_target = self.target is not None
        has_type_feature = self.type is not None and self.feature is not None

        if has_target and has_type_feature:
            raise ValueError(
                "Cannot specify both 'target' and 'type'+'feature'. "
                "Use 'target' for labels or 'type'+'feature' for node/edge features."
            )

        if not has_target and not has_type_feature:
            raise ValueError(
                "Must specify either 'target' (for labels) or both 'type' and 'feature' (for node/edge features)."
            )

        # validate target format (if specified)
        if has_target:
            if not isinstance(self.target, str):
                raise TypeError(f"Target must be str. Got {type(self.target).__name__}")
            if self.target != "y":
                raise ValueError(f"Target must be 'y' for labels. Got: '{self.target}'")

        # validate type + feature format (if specified)
        if has_type_feature:
            if self.type is None or self.feature is None:
                raise ValueError("Both 'type' and 'feature' must be specified together.")

            if not isinstance(self.feature, str):
                raise TypeError(f"Feature must be str. Got {type(self.feature).__name__}")

    @property
    def scaler_target(self) -> str | tuple:
        """Get target in format expected by Scaler class.

        Normalizes type specifications for PyTorch Geometric compatibility:
        - Single-element lists/tuples become strings (node types)
        - Multi-element lists become tuples (edge types)
        - Strings remain as-is
        """
        # return target directly if specified
        if self.target is not None:
            return self.target

        # convert type + feature to (type_spec, feature) tuple
        assert self.type is not None and self.feature is not None

        # Normalize type_spec for PyG HeteroData indexing
        if isinstance(self.type, list):
            # Single-element list -> string (node type)
            if len(self.type) == 1:
                type_spec = self.type[0]
            # Multi-element list -> tuple (edge type)
            else:
                type_spec = tuple(self.type)
        else:
            # Already a string
            type_spec = self.type

        return (type_spec, self.feature)


@dataclass(frozen=True)
class TransformsConfig:
    """Configuration for all transforms (pre and post)."""

    pre: tuple[BuilderTransformConfig, ...] = field(default_factory=tuple)
    post: tuple[BuilderTransformConfig | ScalerTransformConfig, ...] = field(default_factory=tuple)


@dataclass
class Config:
    """
    Root configuration object.

    This is the main entry point for all configuration. It holds
    all sub-configurations as typed, validated dataclasses.
    """

    model: ModelConfigWithArchitecture = field(default_factory=ModelConfigWithArchitecture)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)

    # keep raw config for MLflow logging
    _raw: dict = field(default_factory=dict, repr=False)
    _raw_model: dict = field(default_factory=dict, repr=False)

    @property
    def raw_config(self) -> dict:
        """Get raw configuration dictionary for logging."""
        return self._raw

    @property
    def raw_model(self) -> dict:
        """Get raw model configuration dictionary for logging."""
        return self._raw_model


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
    - Dynamic loading of model-specific architecture configs
    """

    # Registry mapping model types to their config classes
    MODEL_CONFIG_CLASSES: dict[str, type[ModelArchitectureConfig]] = {}

    @classmethod
    def register_config_class(cls, model_type: str):
        """Decorator to register a model architecture config class.

        Args:
            model_type: Model type identifier (e.g., 'HetGAT').

        Returns:
            Decorator function.

        Example:
            @ConfigLoader.register_config_class('HetGAT')
            @dataclass(frozen=True)
            class HetGATArchitectureConfig:
                ...
        """

        def decorator(config_class):
            if model_type in cls.MODEL_CONFIG_CLASSES:
                raise ValueError(f"Config class for '{model_type}' is already registered.")
            cls.MODEL_CONFIG_CLASSES[model_type] = config_class
            return config_class

        return decorator

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

        # parse model name (single source of truth for model identification)
        model_name = raw_config.get("model", {}).get("name", "HetGAT")

        # resolve and load model config path
        if model_config_path is None:
            model_config_path = config_path.parent / f"conf_model_{model_name}.yaml"

        if not model_config_path.exists():
            raise FileNotFoundError(
                f"Model config file not found: {model_config_path}. "
                f"Expected config for model '{model_name}'."
            )

        with open(model_config_path) as f:
            model_raw_config = yaml.safe_load(f)

        # parse main config sections
        training = cls._parse_training_config(raw_config.get("training", {}))
        optimizer = cls._parse_optimizer_config(raw_config.get("optimizer", {}))
        scheduler = cls._parse_scheduler_config(raw_config.get("scheduler", {}))
        early_stopping = cls._parse_early_stopping_config(raw_config.get("early_stopping", {}))
        loss = cls._parse_loss_config(raw_config.get("loss", {}))
        dataset = cls._parse_dataset_config(raw_config.get("dataset", {}))
        model = cls._parse_model_config(
            model_raw_config.get("type", ""), model_raw_config.get("architecture", {})
        )

        # parse transforms
        model_pre, model_post = cls._parse_transforms_config(model_raw_config.get("transforms", {}))
        target_pre, target_post = cls._parse_transforms_config(
            raw_config.get("training", {}).get("target", {})
        )

        # combine target transforms with model transforms
        transforms = TransformsConfig(
            pre=model_pre + target_pre,
            post=model_post + target_post,
        )

        return Config(
            model=model,
            training=training,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            loss=loss,
            dataset=dataset,
            transforms=transforms,
            _raw=raw_config,
            _raw_model=model_raw_config,
        )

    @classmethod
    def _parse_model_config(
        cls,
        model_type: str,
        architecture_data: dict,
    ) -> ModelConfigWithArchitecture:
        """Parse model architecture configuration.

        Args:
            model_type: Model type identifier.
            architecture_data: Architecture configuration dictionary.

        Returns:
            ModelConfigWithArchitecture instance.

        Raises:
            ValueError: If model type not registered or architecture invalid.
        """
        if model_type not in cls.MODEL_CONFIG_CLASSES:
            raise ValueError(
                f"No config class registered for model type '{model_type}'. "
                f"Available types: {list(cls.MODEL_CONFIG_CLASSES.keys())}"
            )

        config_class = cls.MODEL_CONFIG_CLASSES[model_type]
        try:
            architecture = config_class.from_dict(architecture_data)
            if hasattr(architecture, "validate"):
                architecture.validate()
        except Exception as e:
            raise ValueError(
                f"Failed to load architecture for model type '{model_type}': {e}"
            ) from e

        return ModelConfigWithArchitecture(
            type=model_type,
            architecture=architecture,
        )

    @classmethod
    def _parse_transforms_config(
        cls, transforms_data: dict
    ) -> tuple[
        tuple[BuilderTransformConfig, ...],
        tuple[BuilderTransformConfig | ScalerTransformConfig, ...],
    ]:
        """Parse transforms configuration (pre and post).

        Args:
            transforms_data: Transforms configuration dictionary.

        Returns:
            Tuple of (pre_transforms, post_transforms).
        """
        pre = cls._parse_transform_list(transforms_data.get("pre") or [])
        post = cls._parse_transform_list(transforms_data.get("post") or [])
        return pre, post

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
            weight_decay=data.get("weight_decay", 0.0),
            amsgrad=data.get("amsgrad", False),
        )

    @classmethod
    def _parse_scheduler_config(cls, data: dict) -> SchedulerConfig:
        """Parse scheduler configuration section."""
        return SchedulerConfig(
            type=data.get("type", "reduce_on_plateau").lower(),
            # ReduceLROnPlateau
            factor=data.get("factor", 0.1),
            patience=data.get("patience", 10),
            # OneCycleLR
            max_lr=data.get("max_lr", 0.01),
            epochs=data.get("epochs"),
            steps_per_epoch=data.get("steps_per_epoch"),
            pct_start=data.get("pct_start", 0.3),
            anneal_strategy=data.get("anneal_strategy", "cos"),
            # CosineAnnealingWarmRestarts
            T_0=data.get("T_0", 10),
            T_mult=data.get("T_mult", 1),
            eta_min=data.get("eta_min", 0.0),
        )

    @classmethod
    def _parse_early_stopping_config(cls, data: dict) -> EarlyStoppingConfig:
        """Parse early stopping configuration section."""
        return EarlyStoppingConfig(
            enabled=data.get("enabled", False),
            patience=data.get("patience", 50),
            min_delta=data.get("min_delta", 0.0),
            mode=data.get("mode", "min"),
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
    def _parse_transform_list(
        cls, items: list[dict]
    ) -> tuple[BuilderTransformConfig | ScalerTransformConfig, ...]:
        """Parse a list of transforms (builders or scalers).

        Args:
            items: List of transform dictionaries.

        Returns:
            Tuple of parsed transform configs.

        Raises:
            ValueError: If transform structure is invalid.
        """
        configs: list[BuilderTransformConfig | ScalerTransformConfig] = []

        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Transform must be dict, got {type(item).__name__}")

            if "builder" in item:
                configs.append(BuilderTransformConfig(builder=item["builder"]))
            elif "scaler" in item:
                scaler_data = item["scaler"]
                if not isinstance(scaler_data, dict):
                    raise ValueError(f"Scaler data must be dict, got {type(scaler_data).__name__}")

                transform_type = scaler_data.get("transform")
                if not transform_type:
                    raise ValueError("Scaler must have 'transform' key")

                # extract target/type/feature
                target_value = scaler_data.get("target")
                type_value = scaler_data.get("type")
                feature_value = scaler_data.get("feature")

                # extract kwargs (exclude reserved keys)
                kwargs = {
                    k: v
                    for k, v in scaler_data.items()
                    if k not in ("transform", "target", "type", "feature")
                }

                configs.append(
                    ScalerTransformConfig(
                        transform=transform_type,
                        target=target_value,
                        type=type_value,
                        feature=feature_value,
                        kwargs=kwargs,
                    )
                )
            else:
                raise ValueError(f"Unknown transform type: {list(item.keys())}")

        return tuple(configs)


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
