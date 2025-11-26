from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Self

import mlflow
import numpy as np
import torch
import yaml
from mlflow.entities import ViewType


def get_tracking_config() -> dict[str, str]:
    """
    Load MLflow tracking configuration from conf_mlflow.yaml.

    Returns:
        Dictionary with 'tracking_uri' and 'experiment' keys.
    """
    config_path = Path(__file__).parent / "conf_mlflow.yaml"
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
        self.model_dir: Path = self.download_path / experiment.name / self.run_id
        self.module_prefix: str = f"mlstatic_{self.run_id}"

        self.model_module: Any = None
        self.data_module: Any = None
        self.config_module: Any = None

        self._model: Any = None
        self._config: Any = None
        self._dataset: Any = None

        # required files lists
        self.config_files = ["conf_run.yaml"]
        self.module_files = ["model.py", "data.py", "config.py", "model.pt"]
        self.index_files = ["train_indices.txt", "test_indices.txt", "val_indices.txt"]

    def __enter__(self) -> Self:
        """Download artifacts and load all modules."""
        # download artifacts
        self._download_artifacts()

        # validate required files
        self._validate_required_files(self.config_files, self.module_files, self.index_files)

        # load all modules
        print("--- Loading modules...")
        code_path = self.model_dir / "code"

        try:
            self.model_module = self._import_module_from_path(
                f"{self.module_prefix}.model", code_path / "model.py"
            )
            self.data_module = self._import_module_from_path(
                f"{self.module_prefix}.data", code_path / "data.py"
            )
            self.config_module = self._import_module_from_path(
                f"{self.module_prefix}.config", code_path / "config.py"
            )
        except Exception:
            # clean up any partially loaded modules on failure
            self._cleanup_modules()
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up: remove loaded modules from sys.modules."""
        if self.module_prefix is not None:
            self._cleanup_modules()
            self._cleanup_data()

        if exc_type is not None:
            print(f"Error occurred: {exc_type.__name__}: {exc_val}")

    def _download_artifacts(self) -> None:
        """
        Download all necessary artifacts from the run.

        Returns:
            Path to the downloaded model directory.
        """
        # check if already downloaded
        if self.model_dir.exists():
            if self.force:
                shutil.rmtree(self.model_dir)
            else:
                print(f"Model already downloaded at: {self.model_dir}")
                return

        # create directory
        self.model_dir.mkdir(parents=True)

        print(f"Downloading artifacts for run {self.run_id}...")

        # download all artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            # download artifacts to temp directory
            artifact_path = mlflow.artifacts.download_artifacts(  # type: ignore
                run_id=self.run_id, artifact_path="", dst_path=tmpdir
            )

            # copy to final destination
            artifact_path = Path(artifact_path)

            # copy code directory
            if (artifact_path / "code").exists():
                shutil.copytree(artifact_path / "code", self.model_dir / "code")
            else:
                raise FileNotFoundError(f"No 'code' artifacts found for run {self.run_id}")

            # copy split_indices if exists
            if (artifact_path / "split_indices").exists():
                shutil.copytree(artifact_path / "split_indices", self.model_dir / "split_indices")
            else:
                raise FileNotFoundError(f"No 'split_indices' artifacts found for run {self.run_id}")

        print(f"Model downloaded successfully to: {self.model_dir}")

    def _load_split_indices(self) -> dict[str, list]:
        """
        Load data split indices from the split_indices directory.

        Returns:
            Dictionary with split names as keys and list of indices as values.
        """
        split_path = self.model_dir / "split_indices"

        split_indices = {}
        for split_name in ["train", "test", "val"]:
            split_file = split_path / f"{split_name}_indices.txt"
            split_indices[split_name] = np.loadtxt(split_file, dtype=int).tolist()

        return split_indices

    def _validate_required_files(
        self, config_files: list[str], module_files: list[str], index_files: list[str]
    ) -> None:
        """
        Validate that required files exist in the code directory.

        Args:
            config_files: List of configuration filenames to check.
            module_files: List of module filenames to check.
            index_files: List of index filenames to check.

        Raises:
            FileNotFoundError: If any required file is missing.
        """
        code_path = self.model_dir / "code"
        if not code_path.exists():
            raise FileNotFoundError(f"Code directory not found at {code_path}")

        missing = [f for f in config_files if not (code_path / f).exists()]
        if missing:
            missing = ", ".join(missing)
            raise FileNotFoundError(f"Missing required config files in {code_path}: {missing}")

        missing = [f for f in module_files if not (code_path / f).exists()]
        if missing:
            missing = ", ".join(missing)
            raise FileNotFoundError(f"Missing required code files in {code_path}: {missing}")

        index_path = self.model_dir / "split_indices"
        if not index_path.exists():
            raise FileNotFoundError(f"Indices directory not found at {index_path}")

        missing = [f for f in index_files if not (index_path / f).exists()]
        if missing:
            missing = ", ".join(missing)
            raise FileNotFoundError(f"Missing required index files in {index_path}: {missing}")

    def _import_module_from_path(self, module_name: str, file_path: Path) -> Any:
        """
        Dynamically import a Python module from a file path.

        Args:
            module_name: Name to assign to the module.
            file_path: Path to the Python file.

        Returns:
            The imported module.

        Raises:
            ImportError: If module cannot be loaded.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Module file not found: {file_path}")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # clean up on failure
            del sys.modules[module_name]
            raise ImportError(
                f"Failed to execute module {module_name} from {file_path}: {e}"
            ) from e

        return module

    def _cleanup_modules(self) -> None:
        """
        Remove all loaded modules from sys.modules.
        """
        if self.module_prefix is None:
            return
        modules_to_remove = [name for name in sys.modules if name.startswith(self.module_prefix)]
        for module_name in modules_to_remove:
            del sys.modules[module_name]

        self.model_module = None
        self.data_module = None
        self.config_module = None

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
            code_path = self.model_dir / "code"
            print("--- Loading configuration")

            self._config = self.config_module.Config(code_path / self.config_files[0])

        return self._config

    @property
    def model(self) -> tuple[torch.nn.Module]:
        """
        Build the model using the downloaded artifacts.

        Returns:
            The loaded (cpu) PyTorch model ready for inference.
        """

        if not self._model:
            code_path = self.model_dir / "code"
            print("--- Loading model")

            # verify GNN class exists
            if not hasattr(self.model_module, "GNN"):
                raise AttributeError(
                    f"Module {self.model_module.__file__} does not define 'GNN' class"
                )

            # load model using the classmethod
            self._model = self.model_module.GNN.from_checkpoint(self.config, code_path / "model.pt")

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

            self._dataset = self.data_module.STADataset.from_config(self.config)

        return self._dataset
