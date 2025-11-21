from pathlib import Path

import click
import mlflow
import numpy as np
import torch
import torch_geometric as pg
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ml_static.config import Config
from ml_static.data import STADataset
from ml_static.model import GNN
from ml_static.tracker import MLflowtracker
from ml_static.training import train, validate


def run_training(config: Config, check_run: bool = False) -> None:
    """
    Execute training run.

    Args:
        config: Configuration object containing the training parameters.
        sample_run: Bool defining whether to run a "check" run (i.e. using only
            one data sample, to check if the model converges or not)

    """

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    pg.seed_everything(config.seed)

    # load dataset
    dataset = STADataset(config.dataset_path)

    # train/val/test split
    tt = np.random.randn(len(dataset))
    tt_train = tt < 0.7
    tt_val = (tt >= 0.7) & (tt < 0.9)
    tt_test = tt >= 0.9

    train_dataset = dataset[tt_train]
    val_dataset = dataset[tt_val]
    test_dataset = dataset[tt_test]

    train_dataset = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_dataset = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    data_sample = next(iter(train_dataset))

    # define data iterator
    if check_run:
        # if running a check run, use only the first data sample
        data_iterator = [data_sample]
        run_description = "Check run (overfitting)"
    else:
        # for a full run, iterate through all dataset batches
        data_iterator = train_dataset
        run_description = "Training"

    # define model
    model = GNN(data_sample, config.hidden_channels, config.output_channels).to(device)

    # define loss and optimizer
    loss = config.get_loss_function()
    optimizer = config.get_optimizer(model.parameters())

    # define training target
    target_getter = config.get_target_extractor()

    # define training epochs
    epochs = config.epochs

    # set up mlflow tracker
    tracker = MLflowtracker()

    # start mlflow run
    with mlflow.start_run():
        mlflow.log_params(config.raw_config)
        mlflow.log_param("seed", config.seed)

        # log split indices via tracker
        tracker.log_split_indices(
            np.where(tt_train)[0],
            np.where(tt_val)[0],
            np.where(tt_test)[0],
        )

        for epoch in tqdm(range(1, epochs + 1), desc=run_description):
            e_train_loss = 0.0
            e_val_loss = 0.0
            train_batches = 0
            val_batches = 0

            for data in data_iterator:
                data = data.to(device)
                train_loss = train(model, optimizer, loss, data, target_getter)

                e_train_loss += train_loss
                train_batches += 1

            for data in val_dataset:
                data = data.to(device)
                val_loss = validate(model, loss, data, target_getter)
                e_val_loss += val_loss
                val_batches += 1

            # average loss
            e_train_loss /= train_batches
            e_val_loss /= val_batches

            # log metrics
            tracker.log_epoch(epoch, e_train_loss, e_val_loss)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} - Train Loss: {e_train_loss:.4f} - Val Loss: {e_val_loss:.4f}"
                )

        # test phase
        test_loss = 0.0
        test_batches = 0
        for data in test_dataset:
            data = data.to(device)
            loss_out = validate(model, loss, data, target_getter)
            test_loss += loss_out
            test_batches += 1

        test_loss /= test_batches
        mlflow.log_metric("test_loss", test_loss)
        print(f"Test Loss: {test_loss:.4f}")

        tracker.log_training_curves()
        tracker.log_model(model, config.model_name, data)

        return model, dataset, tt_train, tt_val, tt_test


# with open(Path(__file__).parent / "run_params.yaml") as f:
#     params = yaml.safe_load(f)
@click.command("train")
@click.option(
    "-c",
    "--config",
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
    config: str,
    check_run: bool = False,
) -> None:
    """
    Train a GNN model on static traffic assignment data.
    """
    print("--- Training GNN Model ---")

    # set up configuration path
    if config is None:
        config_path = Path(__file__).parent / "conf_run.yaml"
    else:
        config_path = Path(config)

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
        run_training(config, check_run=check_run)
        print("--- Training Complete ---")
    except Exception as e:
        raise Exception(f"Training failed. An unexpected error occurred: {e}") from e


# run script if not called from cli entrypoint
if __name__ == "__main__":
    train_model()
