from pathlib import Path
import click

import mlflow
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ml_static.data import STADataset
from ml_static.model import GNN
from ml_static.training import train, validate
from ml_static.tracker import MLflowtracker
from ml_static.config import Config


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

    # load dataset
    dataset = STADataset(config.dataset_path)
    dataset = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    data_sample = next(iter(dataset))

    # define data iterator
    if check_run:
        # if running a check run, use only the first data sample
        data_iterator = [data_sample]
        run_description = "Check run (overfitting)"
    else:
        # for a full run, iterate through all dataset batches
        data_iterator = dataset
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

        for epoch in tqdm(range(1, epochs + 1), desc=run_description):
            e_train_loss = 0.0
            e_val_loss = 0.0
            num_batches = 0

            for data in data_iterator:
                data = data.to(device)
                train_loss = train(model, optimizer, loss, data, target_getter)
                val_loss = validate(model, loss, data, target_getter)

                e_train_loss += train_loss
                e_val_loss += val_loss
                num_batches += 1

            # average loss
            e_train_loss /= num_batches
            e_val_loss /= num_batches

            # log metrics
            tracker.log_epoch(epoch, e_train_loss, e_val_loss)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} - Train Loss: {e_train_loss:.4f} - Val Loss: {e_val_loss:.4f}"
                )

        tracker.log_training_curves()
        tracker.log_model(model, config.model_name, data)


# with open(Path(__file__).parent / "run_params.yaml") as f:
#     params = yaml.safe_load(f)
@click.command("train")
@click.option(
    "-c",
    "--config",
    default=None,
    help="Path to YAML configuration file. Defaults to run_params.yaml.",
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
