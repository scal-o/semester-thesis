import random
from pathlib import Path

import click
import mlflow
import torch
import torch_geometric as pg
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ml_static.config import Config, load_config
from ml_static.data import DatasetSplit
from ml_static.early_stopping import EarlyStopping
from ml_static.factories import create_optimizer, create_scheduler
from ml_static.losses import LossWrapper
from ml_static.models import model_factory
from ml_static.tracker import MLflowtracker
from ml_static.training import run_epoch, run_test
from ml_static.utils import get_project_root


def run_training(config: Config, check_run: bool = False) -> tuple:
    """
    Execute training run.

    Args:
        config: Configuration object containing the training parameters.
        check_run: Bool defining whether to run a "check" run (i.e. using only
            one data sample, to check if the model converges or not)

    Returns:
        Tuple of (model, dataset_split)
    """

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    seed = config.training.seed if config.training.seed is not None else random.randint(0, 10000)
    pg.seed_everything(seed)

    # create dataset splits with transforms from config
    dataset_split = DatasetSplit.from_config(config, force_reload=False)

    # create dataloaders
    train_loader = DataLoader(
        dataset_split["train"], batch_size=config.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset_split["val"], batch_size=len(dataset_split["val"]), shuffle=False
    )
    test_loader = DataLoader(
        dataset_split["test"], batch_size=len(dataset_split["test"]), shuffle=False
    )

    data_sample = next(iter(train_loader))

    # define data iterator
    if check_run:
        # if running a check run, use only the first data sample
        data_iterator = [data_sample]
        run_description = "Check run (overfitting)"
    else:
        # for a full run, iterate through all dataset batches
        data_iterator = train_loader
        run_description = "Training"

    # define model
    model = model_factory(config).to(device)

    # define loss and optimizer
    loss = LossWrapper.from_config(config)
    loss.register_transform(dataset_split.transform)
    optimizer = create_optimizer(config.optimizer, model.parameters())

    scheduler_runtime_params = {
        "epochs": config.training.epochs,
        "steps_per_epoch": len(train_loader),
    }
    print(scheduler_runtime_params)
    scheduler = create_scheduler(config.scheduler, optimizer, scheduler_runtime_params)

    # initialize early stopping if enabled
    early_stopping = None
    if config.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
            mode=config.early_stopping.mode,
        )

    # define training epochs
    epochs = config.training.epochs

    # set up mlflow tracker
    tracker = MLflowtracker()

    # start mlflow run
    with mlflow.start_run():
        tracker.log_configs()
        tracker.log_params(config.raw_config)
        tracker.log_params(config.raw_model)
        tracker.log_seed(seed)

        # log dataset split information
        tracker.log_dataset_info(dataset_split)

        for epoch in tqdm(range(1, epochs + 1), desc=run_description):
            # train/val phase
            e_train_loss, e_val_loss, train_components, val_components = run_epoch(
                model,
                data_iterator,
                val_loader,
                optimizer,
                loss,
                device,
            )

            # step scheduler
            if check_run:
                scheduler.step(e_train_loss)
            else:
                scheduler.step(e_val_loss)

            # get current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # log metrics
            tracker.log_epoch(
                epoch, e_train_loss, e_val_loss, train_components, val_components, current_lr
            )

            # check early stopping
            if early_stopping is not None:
                if early_stopping(e_val_loss, model, epoch):
                    print(
                        f"\nEarly stopping triggered at epoch {epoch}. "
                        f"Best validation loss: {early_stopping.best_metric:.4f} at epoch {early_stopping.best_epoch}"
                    )
                    break

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} - Train Loss: {e_train_loss:.4f} - Val Loss: {e_val_loss:.4f} - LR: {current_lr:.2e}"
                )

                if val_components:
                    comp_str = "Loss components: " + " - ".join(
                        f"{k.replace('unweighted_', '')}: {v:.2f}"
                        for k, v in val_components.items()
                        if k.startswith("unweighted_")
                    )
                    print(comp_str)

        # load best model if early stopping was used
        if early_stopping is not None:
            early_stopping.load_best_model(model)
            tracker.log_best_model_info(early_stopping.best_epoch, early_stopping.best_metric)
            print(
                f"\nLoaded best model from epoch {early_stopping.best_epoch} "
                f"(val_loss: {early_stopping.best_metric:.4f})"
            )

        # test phase
        test_loss = run_test(model, test_loader, loss, device)
        tracker.log_test_loss(test_loss)
        print(f"Test Loss: {test_loss:.4f}")

        tracker.log_training_curves()
        tracker.log_model(model, config.model.type, dataset_split["train"][0])

        print("--- Computing Performance Statistics ---")
        # log performance reports for all dataset splits
        datasets = {
            "train": dataset_split["train"],
            "validation": dataset_split["val"],
            "test": dataset_split["test"],
        }
        stats = tracker.log_all_performance_reports(model, datasets)
        # print stats table
        print(stats)

        print("--- Logging Sample Scenario Predictions ---")
        # log prediction plots for random scenarios from each split
        tracker.log_random_scenario_predictions(model, datasets, num_scenarios=5)

        return model, dataset_split


@click.command("train")
@click.option(
    "-c",
    "--config-path",
    default=None,
    help="Path to YAML configuration file. Defaults to conf_run.yaml.",
)
@click.option(
    "--check-run",
    is_flag=True,
    default=False,
    help="Run a check run on a single data sample to verify model convergence.",
)
def train_model(
    config_path: Path | str | None = None,
    check_run: bool = False,
) -> tuple:
    """
    Train a GNN model on static traffic assignment data.

    Returns:
        Tuple of (model, dataset_split)
    """
    print("--- Training GNN Model ---")

    # set up configuration path
    if config_path is None:
        config_path = get_project_root() / "configs" / "conf_run.yaml"
    else:
        config_path = Path(config_path)

    # check path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # load config
    config = load_config(config_path)
    print(f"Configuration loaded from {config_path}")

    print(f"Dataset: {config.dataset.full_path}")
    print(f"Check run: {check_run}")

    # run training
    try:
        model, dataset_split = run_training(config, check_run=check_run)
        print("--- Training Complete ---")

        return model, dataset_split
    except Exception as e:
        raise Exception(f"Training failed. An unexpected error occurred: {e}") from e


# run script if not called from cli entrypoint
if __name__ == "__main__":
    # When running interactively, we call the underlying function directly
    # to avoid click's command-line handling, which can exit the script.
    model, dataset_split = train_model.callback(config_path=None, check_run=True)

    results = {
        "model": model,
        "dataset_split": dataset_split,
    }
