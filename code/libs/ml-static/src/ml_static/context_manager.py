from __future__ import annotations

import importlib.util
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Self

import mlflow
import torch
import yaml
from mlflow.entities import ViewType

from ml_static.utils import get_project_root

# lock to prevent concurrent sys.path/sys.modules manipulation
_import_lock = threading.RLock()


def get_tracking_config() -> dict[str, str]:
    """
    Load MLflow tracking configuration from conf_mlflow.yaml.

    Returns:
        Dictionary with 'tracking_uri' and 'experiment' keys.
    """
    config_path = get_project_root() / "configs" / "conf_mlflow.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"MLflow configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    tracking_uri = config.get("tracking_uri")
    experiment_name = config.get("experiment")

    if not tracking_uri or not experiment_name:
        raise ValueError(f"'tracking_uri' and 'experiment' must be specified in {config_path}")

    return config


def list_runs(
    config: dict[str, str] | None = None,
    filter_string: str | None = None,
) -> dict[str, str]:
    """
    List all runs for an experiment with their key metrics and parameters.

    Args:
        config: MLflow tracking configuration dictionary. If None, loads from conf_mlflow.yaml.
        filter_string: MLflow filter string (e.g., "metrics.test_loss < 0.1").

    Returns:
        Dictionary mapping run IDs to run names.
    """
    # load defaults from config if not provided
    if not config:
        config = get_tracking_config()

    tracking_uri = config["tracking_uri"]
    experiment_name = config["experiment"]

    # set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # search runs
    runs = mlflow.search_runs(  # type: ignore
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string if filter_string else "",
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    if runs.empty:  # type: ignore
        print(f"No runs found for experiment '{experiment_name}'")
        return dict()

    # clean up the dataframe for better readability
    # keep only the most relevant columns
    relevant_cols = ["run_id", "tags.mlflow.runName", "start_time"]
    runs = runs[relevant_cols]  # type: ignore
    runs = runs.sort_values("start_time", ascending=False)  # type: ignore

    runs = {name: id for id, name in zip(runs["run_id"], runs["tags.mlflow.runName"])}  # type: ignore

    return runs


class RunContext:
    """Context manager for loading and managing model artifacts from MLflow runs."""

    def __init__(
        self,
        run_id: str | None = None,
        config: dict[str, str] | None = None,
        download_path: Path | str = "downloaded_models",
        force: bool = False,
    ):
        """
        Initialize the context for a run.

        Args:
            run_id: MLflow run ID. Defaults to last run if None.
            download_path: Base path for downloaded models.
            tracking_uri: MLflow tracking URI (defaults to conf_mlflow.yaml).
            force: If True, re-download even if already exists.
        """

        # if no config provided, load from file
        self.run_config = get_tracking_config() if config is None else config

        # if no run id provided, load last run
        self.run_id = list(list_runs(self.run_config).values())[0] if run_id is None else run_id

        # set tracking uri
        self.tracking_uri = self.run_config["tracking_uri"]

        # set download path for artifacts
        self.download_path = Path(download_path)

        self.force: bool = force

        # create model directory path
        mlflow.set_tracking_uri(self.tracking_uri)
        run = mlflow.get_run(self.run_id)
        experiment = mlflow.get_experiment(run.info.experiment_id)
        run_name = run.data.tags.get("mlflow.runName", self.run_id)
        self.model_dir: Path = self.download_path / experiment.name / run_name
        self.module_prefix: str = f"mlstatic_{self.run_id}"

        self.models_module: Any = None
        self.data_module: Any = None
        self.config_module: Any = None

        self._model: Any = None
        self._config: Any = None
        self._dataset: Any = None
        self._data_split: Any = None

    def __enter__(self) -> Self:
        """Download artifacts and surgically load modules."""
        self._download_artifacts()

        # The core logic is now much safer and cleaner
        with _import_lock:
            print("starting module load")
            # 1. Save a reference to any currently installed ml_static modules
            self._saved_modules = {}
            for name in list(sys.modules.keys()):
                if name == "ml_static" or name.startswith("ml_static."):
                    self._saved_modules[name] = sys.modules.pop(name)

            try:
                # 2. Define the path to the downloaded package
                code_path = self.model_dir / "code"
                package_init_path = code_path / "__init__.py"

                if not package_init_path.is_file():
                    raise FileNotFoundError(
                        f"The package init file '__init__.py' was not found in {code_path}"
                    )

                # 3. Create a module spec for the downloaded package.
                # This tells the import system that the 'code' directory is the 'ml_static' package.
                spec = importlib.util.spec_from_file_location("ml_static", package_init_path)

                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create a module spec for {package_init_path}")

                # 4. Create the module and add it to sys.modules.
                ml_static_module = importlib.util.module_from_spec(spec)
                sys.modules["ml_static"] = ml_static_module

                # Execute the module code (the __init__.py)
                spec.loader.exec_module(ml_static_module)

                # 5. Now, absolute imports work correctly.
                import ml_static.config as config
                import ml_static.data as data
                import ml_static.models as models

                self.config_module = config
                self.data_module = data
                self.models_module = models

            except Exception:
                # If anything goes wrong, clean up immediately and restore state
                self.__exit__(*sys.exc_info())
                raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up modules and restore the original environment."""
        with _import_lock:
            # 1. Remove all modules that were loaded from the downloaded code
            loaded_code_dir = str(self.model_dir / "code")
            for name, module in list(sys.modules.items()):
                # A module is one of ours if it has a __file__ pointing to our code dir
                if (
                    hasattr(module, "__file__")
                    and module.__file__
                    and str(module.__file__).startswith(loaded_code_dir)
                ):
                    del sys.modules[name]

            # 2. Restore the original modules that were saved in __enter__
            if hasattr(self, "_saved_modules"):
                sys.modules.update(self._saved_modules)
                del self._saved_modules  # Clean up the saved dict to avoid holding refs

        self._cleanup_data()

        if exc_type is not None:
            print(f"Error occurred: {exc_type.__name__}: {exc_val}")

    def _download_artifacts(self) -> None:
        """
        Download all necessary artifacts from the run.
        """
        # check if already downloaded
        if self.model_dir.exists():
            if self.force:
                shutil.rmtree(self.model_dir)
            else:
                print(f"Model already downloaded at: {self.model_dir}")
                return

        # create directory (mlflow will create the final dir, but we need parents)
        self.model_dir.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading artifacts for run {self.run_id}...")

        # download all artifacts directly to the destination directory
        mlflow.artifacts.download_artifacts(  # type: ignore
            run_id=self.run_id, artifact_path="", dst_path=str(self.model_dir)
        )

        print(f"Model downloaded successfully to: {self.model_dir}")

    def _cleanup_data(self) -> None:
        """
        Remove all data instances from class.
        """

        self._model = None
        self._config = None
        self._dataset = None

    @property
    def config(self) -> Any:
        """
        Build the Config instance using the downloaded artifacts.

        Returns:
            Config instance from the run's configuration.
        """

        if not self._config:
            config_path = self.model_dir / "configs" / "conf_run.yaml"
            print("--- Loading configuration")

            self._config = self.config_module.load_config(config_path)

        return self._config

    @property
    def model(self) -> torch.nn.Module:
        """
        Build the model using the downloaded artifacts.

        Returns:
            The loaded (cpu) PyTorch model ready for inference.
        """

        if not self._model:
            print("--- Loading model")

            # load checkpoint to get model type and state
            checkpoint_path = self.model_dir / "model" / "model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model_type = checkpoint.get("model_type")

            if not model_type:
                raise ValueError("Checkpoint does not contain 'model_type' field")

            # verify factory exists
            if not hasattr(self.models_module, "model_factory"):
                raise AttributeError("Models module does not define 'model_factory'")

            # create model from config and load checkpoint
            self._model = self.models_module.model_factory(self.config)
            self._model.load_state_dict(checkpoint["state_dict"])

        return self._model

    @property
    def dataset(self) -> Any:
        """
        Build a dataset using the correct STADataset class from the run.

        Returns:
            STADataset instance compatible with the model.
        """

        if not self._dataset:
            print("--- Loading dataset")

            # verify STADataset exists
            if not hasattr(self.data_module, "STADataset"):
                raise AttributeError("Data module does not define 'STADataset' class")

            self._dataset = self.data_module.STADataset.from_config(self.config, force_reload=True)

        return self._dataset

    @property
    def data_split(self) -> Any:
        """
        Create DatasetSplit using the seed from the run configuration.
        Splits are deterministically regenerated from the seed.

        Returns:
            DatasetSplit instance with train/val/test splits.
        """

        if not self._data_split:
            print("--- Creating data split from config")

            # verify DatasetSplit exists
            if not hasattr(self.data_module, "DatasetSplit"):
                raise AttributeError("Data module does not define 'DatasetSplit' class")

            self._data_split = self.data_module.DatasetSplit.from_config(
                self.config, force_reload=True
            )

        return self._data_split
