import os
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
import torch_geometric
import yaml

# env vars to load
ENV_VARS = {
    # force mlflow to use uv to register all deps
    "MLFLOW_LOCK_MODEL_DEPENDENCIES": "true",
}

# load env vars
env = os.environ
for key, value in ENV_VARS.items():
    env[key] = str(value)


# create tracking class
class MLflowtracker:
    def __init__(self):
        # retrieve configs
        # run_conf contains mlflow configs (tracking uri and experiment name)
        # run_params contains model parameters (name, epochs, loss, optimizer)
        with open(Path(__file__).parent / "conf_mlflow.yaml") as f:
            config = yaml.safe_load(f)

        # set uri and experiment
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(config["experiment"])

        # create losses lists
        self.train_losses = list()
        self.val_losses = list()

    def log_epoch(self, epoch, train_loss, val_loss):
        """Log metrics for a single epoch"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            step=epoch,
        )

    def log_training_curves(self):
        """Generate and log training curves"""
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        # loss subplot
        ax.plot(self.train_losses, label="Training loss")
        ax.plot(self.val_losses, label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss curves")
        ax.legend()

        # save and log plot
        mlflow.log_figure(fig, "plots/training_curves.png")
        plt.close(fig)

    def log_model(self, model, model_name, data):
        """
        Custom method to log model with related artifacts:
        - model weights (saved via pytorch geometric )
        - source code (run.py + model.py)
        """

        code_dir = Path(__file__).parent

        with tempfile.TemporaryDirectory() as dirname:
            tempdir = Path(dirname)

            # save source code and config
            # TODO: if passing custom config to the cli, copy that instead of the default
            shutil.copytree(code_dir, tempdir, dirs_exist_ok=True)

            # save model weights
            torch.save(model.state_dict(), tempdir / "model.pt")

            # save model summary
            with open(tempdir / "model_summary.txt", "w") as f:
                model_summary = torch_geometric.nn.summary(model, data)
                f.write(model_summary)

            # log artifacts
            mlflow.log_artifacts(tempdir, artifact_path="code")
