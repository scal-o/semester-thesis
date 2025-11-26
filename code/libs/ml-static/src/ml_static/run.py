from pathlib import Path

import click
import mlflow
import numpy as np
import torch
import torch_geometric as pg
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ml_static.config import Config
from ml_static.data import STADataset, VarTransform, create_splits
from ml_static.model import GNN
from ml_static.tracker import MLflowtracker
from ml_static.training import run_epoch, run_test


def run_training(config: Config, check_run: bool = False) -> tuple:
    """
    Execute training run.

    Args:
        config: Configuration object containing the training parameters.
        check_run: Bool defining whether to run a "check" run (i.e. using only
            one data sample, to check if the model converges or not)

    Returns:
        Tuple of (model, dataset, train_loader, val_loader, test_loader, device, target_getter, tracker, tt_train, tt_val, tt_test)
    """

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    pg.seed_everything(config.seed)

    # load dataset (and implicitly apply transforms) from config
    dataset = STADataset.from_config(config)

    # train/val/test split
    (
        (train_split, val_split, test_split),
        (tt_train, tt_val, tt_test),
    ) = create_splits(dataset, (0.7, 0.15, 0.15))

    # create dataloaders
    train_loader = DataLoader(train_split, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_split, batch_size=len(val_split), shuffle=False)
    test_loader = DataLoader(test_split, batch_size=len(test_split), shuffle=False)

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
    model = GNN.from_config(config).to(device)

    # define loss and optimizer
    loss = config.get_loss_function()
    optimizer = config.get_optimizer(model.parameters())
    scheduler = config.get_scheduler(optimizer)

    # define training epochs
    epochs = config.epochs

    # set up mlflow tracker
    tracker = MLflowtracker()

    # start mlflow run
    with mlflow.start_run():
        tracker.log_params(config.raw_config)
        tracker.log_seed(config.seed)

        # log split indices via tracker
        tracker.log_split_indices(tt_train, tt_val, tt_test)

        for epoch in tqdm(range(1, epochs + 1), desc=run_description):
            # train/val phase
            e_train_loss, e_val_loss = run_epoch(
                model,
                data_iterator,
                val_loader,
                optimizer,
                loss,
                device,
            )

            # step scheduler
            scheduler.step(e_val_loss)

            # get current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # log metrics
            tracker.log_epoch(epoch, e_train_loss, e_val_loss, current_lr)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} - Train Loss: {e_train_loss:.4f} - Val Loss: {e_val_loss:.4f} - LR: {current_lr:.2e}"
                )

        # test phase
        test_loss = run_test(model, test_loader, loss, device)
        tracker.log_test_loss(test_loss)
        print(f"Test Loss: {test_loss:.4f}")

        tracker.log_training_curves()
        tracker.log_model(model, config.model_name, data_sample)

        print("--- Computing Performance Statistics ---")
        # log performance reports for all dataset splits
        loaders = {
            "train": train_loader,
            "validation": val_loader,
            "test": test_loader,
        }
        stats = tracker.log_all_performance_reports(model, loaders)
        # print stats table
        print(stats)

        return (
            model,
            dataset,
            tt_train,
            tt_val,
            tt_test,
        )


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
        Tuple of (model, dataset, tt_train, tt_val, tt_test)
    """
    print("--- Training GNN Model ---")

    # set up configuration path
    if config_path is None:
        config_path = Path(__file__).parent / "conf_run.yaml"
    else:
        config_path = Path(config_path)

    # check path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # load config
    config = Config(config_path)
    print(f"Configuration loaded from {config_path}")

    print(f"Dataset: {config.dataset_path}")
    print(f"Check run: {check_run}")

    # run training
    try:
        (
            model,
            dataset,
            tt_train,
            tt_val,
            tt_test,
        ) = run_training(config, check_run=check_run)
        print("--- Training Complete ---")

        return model, dataset, tt_train, tt_val, tt_test
    except Exception as e:
        raise Exception(f"Training failed. An unexpected error occurred: {e}") from e


# run script if not called from cli entrypoint
if __name__ == "__main__":
    # When running interactively, we call the underlying function directly
    # to avoid click's command-line handling, which can exit the script.
    model, dataset, tt_train, tt_val, tt_test = train_model.callback(
        config_path=None, check_run=True
    )

    results = {
        "model": model,
        "dataset": dataset,
        "tt_train": tt_train,
        "tt_val": tt_val,
        "tt_test": tt_test,
    }
