"""
Base class for sequential transform pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from torch_geometric.transforms import BaseTransform

from ml_static.config import BuilderTransformConfig, ScalerTransformConfig
from ml_static.transforms.builders import BuilderTransform
from ml_static.transforms.scalers import FeatureScaler

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import Config
    from ml_static.data import STADataset


class SequentialTransform(BaseTransform):
    """
    Base class for sequential transform pipelines.

    Applies a list of transforms in sequence, with a unified interface
    for both forward() and fit_dataset() operations.
    """

    def __init__(self, transforms: list) -> None:
        """
        Initialize with a list of transforms.

        Args:
            transforms: List of BaseTransform instances to apply in sequence.
        """
        self.transforms = transforms

    @classmethod
    def from_config(cls, config: Config, stage: str) -> Self:
        """
        Create SequentialTransform from configuration.

        Args:
            config: Root configuration object.
            stage: Stage of the pipeline ('pre' or 'post').

        Returns:
            SequentialTransform instance.
        """

        transforms = []

        # select transform configurations based on stage
        if stage == "pre":
            iterator = config.transforms.pre
        elif stage == "post":
            iterator = config.transforms.post
        else:
            raise ValueError(f"Invalid transform stage: {stage}. Must be 'pre' or 'post'.")

        # build transforms from configurations using cascaded from_config
        for transform_cfg in iterator:
            if isinstance(transform_cfg, BuilderTransformConfig):
                transforms.append(BuilderTransform.from_config(transform_cfg))
            elif isinstance(transform_cfg, ScalerTransformConfig):
                transforms.append(FeatureScaler.from_config(transform_cfg))
            else:
                raise ValueError(f"Unknown transform config type: {type(transform_cfg)}")

        return cls(transforms=transforms)

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Apply all transforms sequentially.

        Args:
            data: Input HeteroData object.

        Returns:
            Transformed HeteroData with all transforms applied.
        """
        for transform in self.transforms:
            data = transform(data)
        return data

    def fit_dataset(self, dataset: STADataset) -> None:
        """
        Fit all transforms on the dataset.

        Only transforms that need fitting (e.g., scalers) will perform operations.
        Stateless transforms (e.g., builders) will be skipped.

        Args:
            dataset: Dataset to fit transforms on.
        """
        for transform in self.transforms:
            transform.fit_dataset(dataset)
