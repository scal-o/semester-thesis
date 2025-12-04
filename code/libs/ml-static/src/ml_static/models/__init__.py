from ml_static.models.factory import model_factory, register_model

# Import model implementations to trigger registration
from ml_static.models.HetGAT import HetGAT, HetGATArchitectureConfig

# Export public API
__all__ = ["model_factory", "register_model", "HetGAT", "HetGATArchitectureConfig"]
