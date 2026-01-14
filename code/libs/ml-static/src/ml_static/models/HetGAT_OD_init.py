"""
HetGAT with OD-based node initialization.

Instead of preprocessing existing node features, this model initializes
node embeddings from OD demand using virtual edges.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import torch
import torch.nn as nn

from ml_static.config import ConfigLoader
from ml_static.models import register_model
from ml_static.models.components import (
    edge_processors,
    encoders,
    od_initializers,
    predictors,
    preprocessors,
)

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import Config


# =============================================================================
# Configuration for HetGAT_OD_init
# =============================================================================


@ConfigLoader.register_config_class("HetGAT_OD_init")
@dataclass(frozen=True)
class HetGAT_OD_init_ArchitectureConfig:
    """HetGAT with OD-based node initialization.

    Attributes:
        edge_processor: LinearEdgeProcessorConfig or RbfEdgeProcessorConfig.
        preprocessor: Node preprocessor configuration.
        encoders: Tuple of encoder configurations.
        predictor: Edge predictor configuration.
    """

    edge_processor: (
        edge_processors.LinearEdgeProcessorConfig | edge_processors.RbfEdgeProcessorConfig
    )
    preprocessor: preprocessors.PreprocessorConfig
    encoders: tuple[encoders.EncoderConfig, ...]
    predictor: predictors.PredictorConfig

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse HetGAT_OD_init_ArchitectureConfig from dictionary.

        Args:
            data: Dict with keys: edge_processor, encoders, predictor.

        Returns:
            HetGAT_OD_init_ArchitectureConfig instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If values are invalid.
        """
        cls._validate_dict(data)

        # determine edge processor type
        processor_data = data["edge_processor"]
        processor_type = processor_data.get("type", "linear")

        if processor_type == "linear":
            edge_proc = edge_processors.LinearEdgeProcessorConfig.from_dict(processor_data)
        elif processor_type == "rbf":
            edge_proc = edge_processors.RbfEdgeProcessorConfig.from_dict(processor_data)
        else:
            raise ValueError(f"Unknown edge processor type: {processor_type}")

        # parse preprocessor
        preproc = preprocessors.PreprocessorConfig.from_dict(data["preprocessor"])

        # parse encoders
        encoders_list = [encoders.EncoderConfig.from_dict(d) for d in data["encoders"]]
        encoders_list = tuple(encoders_list)

        # parse predictor
        predictor = predictors.PredictorConfig.from_dict(data["predictor"])

        return cls(
            edge_processor=edge_proc,
            preprocessor=preproc,
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
        required = {"edge_processor", "preprocessor", "encoders", "predictor"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in HetGAT_OD_init architecture: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        return asdict(self)

    def validate(self) -> None:
        """Validate HetGAT_OD_init-specific constraints."""
        if len(self.encoders) < 1:
            raise ValueError("HetGAT_OD_init requires at least one encoder")

        # validate edge processor
        self.edge_processor.validate()

        # validate preprocessor
        self.preprocessor.validate()

        # validate all encoders
        for encoder in self.encoders:
            encoder.validate()

        # validate predictor
        self.predictor.validate()


# =============================================================================
# HetGAT_OD_init Model Implementation
# =============================================================================


@register_model("HetGAT_OD_init")
class HetGAT_OD_init(nn.Module):
    """
    HetGAT with OD-based node initialization.

    Flow:
    1. OD Initializer: virtual edges → node embeddings
    2. Encoders: process graph with attention
    3. Predictor: edge predictions
    """

    def __init__(
        self,
        edge_processor: nn.Module,
        od_initializer: nn.Module,
        preprocessor: nn.Module,
        encoders: list[nn.Module],
        predictor: nn.Module,
    ):
        """Initialize HetGAT_OD_init model.

        Args:
            edge_processor: Edge embedding processor (Linear or RBF).
            od_initializer: OD node initializer for aggregation.
            preprocessor: Node feature preprocessor.
            encoders: List of encoder modules.
            predictor: Edge predictor module.
        """
        super().__init__()
        self.edge_processor = edge_processor
        self.od_initializer = od_initializer
        self.preprocessor = preprocessor
        self.encoders = nn.ModuleList(encoders)
        self.predictor = predictor

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Create HetGAT_OD_init model from Config object.

        Args:
            config: Full configuration object.

        Returns:
            Configured HetGAT_OD_init instance.

        Raises:
            ValueError: If no architecture is specified or wrong type.
        """
        arch = config.model.architecture

        if not isinstance(arch, HetGAT_OD_init_ArchitectureConfig):
            raise ValueError(
                f"Expected HetGAT_OD_init_ArchitectureConfig, got {type(arch).__name__}. "
                "Ensure model config file exists and is loaded correctly."
            )

        arch.validate()

        # build edge processor
        if isinstance(arch.edge_processor, edge_processors.LinearEdgeProcessorConfig):
            edge_proc = edge_processors.LinearEdgeProcessor.from_config(arch.edge_processor)
        elif isinstance(arch.edge_processor, edge_processors.RbfEdgeProcessorConfig):
            edge_proc = edge_processors.RbfEdgeProcessor.from_config(arch.edge_processor)
        else:
            raise ValueError(f"Unknown edge processor config type: {type(arch.edge_processor)}")

        # build OD node initializer (no config needed)
        od_init = od_initializers.ODNodeInitializer()

        # build preprocessor
        preproc = preprocessors.NodePreprocessor.from_config(arch.preprocessor)

        # build encoders
        encoders_list = []
        for encoder_config in arch.encoders:
            encoder = encoders.EncoderBase.from_config(encoder_config)
            encoders_list.append(encoder)

        # build predictor
        predictor = predictors.EdgePredictor.from_config(arch.predictor)

        return cls(edge_proc, od_init, preproc, encoders_list, predictor)

    def forward(self, graph: HeteroData):
        """Forward pass through model.

        Args:
            graph: Heterogeneous graph data.

        Returns:
            Edge predictions for real edges.
        """
        # 1. process virtual edge attributes to embeddings
        virtual_edge_type = ("nodes", "virtual", "nodes")
        graph = self.edge_processor(x=None, data=graph, edge_type=virtual_edge_type)

        # 2. aggregate edge embeddings into node features
        graph = self.od_initializer(x=None, data=graph, edge_type=virtual_edge_type)

        # 3. preprocess node features
        g = self.preprocessor(x=None, data=graph, type="nodes")

        # 4. apply encoders
        for encoder in self.encoders:
            g = encoder(g, graph)

        # 5. edge predictions
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
            "model_type": "HetGAT_OD_init",
        }

    @classmethod
    def from_checkpoint(cls, config: Config, checkpoint_path: Path | str) -> Self:
        """Load HetGAT_OD_init model from checkpoint file.

        Args:
            config: Configuration object for model architecture.
            checkpoint_path: Path to checkpoint file (.pt).

        Returns:
            Loaded HetGAT_OD_init model.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            ValueError: If checkpoint is incompatible with model.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # verify model type
        if checkpoint.get("model_type") != "HetGAT_OD_init":
            raise ValueError(
                f"Checkpoint model type '{checkpoint.get('model_type')}' "
                f"does not match HetGAT_OD_init"
            )

        # create model from config
        model = cls.from_config(config)

        # load state dict
        model.load_state_dict(checkpoint["state_dict"])

        return model
