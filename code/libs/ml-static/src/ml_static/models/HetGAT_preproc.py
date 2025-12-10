from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import torch
import torch.nn as nn

from ml_static.config import ConfigLoader
from ml_static.models import register_model
from ml_static.models.components import encoders, predictors, preprocessors
from ml_static.utils.validation import validate_node_attribute

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import Config


# =============================================================================
# Configuration for HetGAT
# =============================================================================


@ConfigLoader.register_config_class("HetGAT_preproc")
@dataclass(frozen=True)
class HetGAT_preproc_ArchitectureConfig:
    """HetGAT-specific architecture configuration.

    Attributes:
        encoders: Tuple of encoder configurations (virtual and real).
        predictor: Edge predictor configuration.
    """

    preprocessor: preprocessors.PreprocessorConfig
    encoders: tuple[encoders.EncoderConfig, ...]
    predictor: predictors.PredictorConfig

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse HetGATArchitectureConfig from dictionary.

        Args:
            data: Dict with keys: preprocessor, encoders, predictor.

        Returns:
            HetGATArchitectureConfig instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If values are invalid.
        """
        cls._validate_dict(data)

        # Parse preprocessor
        preprocessor = preprocessors.PreprocessorConfig.from_dict(data["preprocessor"])

        # Parse encoders
        encoders_list = [encoders.EncoderConfig.from_dict(d) for d in data["encoders"]]
        encoders_list = tuple(encoders_list)

        # Parse predictor
        predictor = predictors.PredictorConfig.from_dict(data["predictor"])

        return cls(
            preprocessor=preprocessor,
            encoders=encoders_list,
            predictor=predictor,
        )

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """Validate that required keys are present in dict.

        Args:
            data: Dictionary to validate.

        Raises:
            KeyError: If required keys are missing.
        """
        required = {"preprocessor", "encoders", "predictor"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in HetGAT architecture: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        return asdict(self)

    def validate(self) -> None:
        """Validate HetGAT-specific constraints."""
        if len(self.encoders) < 1:
            raise ValueError("HetGAT requires at least one encoder")

        # Validate preprocessor
        self.preprocessor.validate()

        # Validate all encoders
        for encoder in self.encoders:
            encoder.validate()

        # Validate predictor
        self.predictor.validate()


# =============================================================================
# HetGAT Model Implementation
# =============================================================================


@register_model("HetGAT_preproc")
class HetGAT_preproc(nn.Module):
    """
    Heterogeneous Graph Attention Network (HetGAT) model.
    Structure from Liu & Meidani, 2024 (adapted).
    Integrates preprocessing fully connected layers to normalize node
    feature dimension.

    This model processes graphs with both virtual and real edges using
    separate attention-based encoders, followed by an MLP for link prediction.
    """

    def __init__(
        self,
        preprocessor: nn.Module,
        encoders: list[nn.Module],
        predictor: nn.Module,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoders = nn.ModuleList(encoders)
        self.predictor = predictor

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Create HetGAT model from Config object.

        Args:
            config: Full configuration object.

        Returns:
            Configured HetGAT instance.

        Raises:
            ValueError: If no architecture is specified or wrong type.
        """
        arch = config.model.architecture

        if not isinstance(arch, HetGAT_preproc_ArchitectureConfig):
            raise ValueError(
                f"Expected HetGATArchitectureConfig, got {type(arch).__name__}. "
                "Ensure model config file exists and is loaded correctly."
            )

        arch.validate()

        preprocessor = preprocessors.NodePreprocessor.from_config(arch.preprocessor)

        encoders_list = []
        for encoder_config in arch.encoders:
            encoder = encoders.EncoderBase.from_config(encoder_config)
            encoders_list.append(encoder)

        predictor = predictors.EdgePredictor.from_config(arch.predictor)

        return cls(preprocessor, encoders_list, predictor)

    def forward(self, graph: HeteroData):
        # preprocess and extract node features
        g = self.preprocessor(None, graph)

        # apply encoders sequentially
        for encoder in self.encoders:
            g = encoder(g, graph)
        # make edge predictions
        z = self.predictor(g, graph)

        return z

    def extract_checkpoint(self) -> dict:
        """Extract model checkpoint for saving.

        Returns:
            Dictionary containing state_dict and model metadata.
            The state_dict is deep-copied to ensure independence from the model.
        """
        return {
            "state_dict": copy.deepcopy(self.state_dict()),
            "model_type": "HetGAT_preproc",
        }

    @classmethod
    def from_checkpoint(cls, config: Config, checkpoint_path: Path | str) -> Self:
        """Load HetGAT model from checkpoint file.

        Args:
            config: Configuration object for model architecture.
            checkpoint_path: Path to checkpoint file (.pt).

        Returns:
            Loaded HetGAT model.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            ValueError: If checkpoint is incompatible with model.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # verify model type
        if checkpoint.get("model_type") != "HetGAT_preproc":
            raise ValueError(
                f"Checkpoint model type '{checkpoint.get('model_type')}' does not match HetGAT_preproc."
            )

        # create model from config
        model = cls.from_config(config)

        # load state dict
        model.load_state_dict(checkpoint["state_dict"])

        return model
