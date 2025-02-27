# train autoencoder
import wandb
import logging
import pandas as pd
from typing import List, Optional

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils.class_weight import compute_class_weight
from omegaconf import OmegaConf

from src.debiasing.models import AdversarialPredictor, MetaDataPrediction, cmmdRegularizedPredictor

LOG = logging.getLogger(__name__)


def adjust_checkpoint_callback(config: Optional[DictConfig]) -> None:
    """
    Adjusts the checkpoint callback to save the best model based on the validation loss.

    Args:
        config (DictConfig): The configuration object.

    Returns:
        DictConfig: The modified configuration object.
    """
    callbacks = config.get("callbacks", None)
    if callbacks:
        if config.predictor.target in ["sex", "site", "age_bin"]:
            target = config.predictor.target
            monitor = f"val/{target}/bacc"
            config.callbacks.model_checkpoint.monitor = monitor
            config.callbacks.model_checkpoint.mode = "max"
            config.callbacks.model_checkpoint.filename = "epoch-{epoch}-val_bacc-{" + f"{monitor}:.2f" + "}"
            LOG.info("Checkpoint callback adjusted to monitor %s", monitor)

def train(config: DictConfig):

    class_weighted = config.get("class_weighted", False)
    csv_path = config.get("csv_path", None)

    if class_weighted and not csv_path:
        LOG.error("Class weights are requested but no CSV file is provided. Exiting.")
        raise ValueError("Class weights are requested but no CSV file is provided.")

    if class_weighted:
        try:
            dataframe = pd.read_csv(csv_path)
            target: str = config.predictor.target
            target = target.capitalize()
            class_weights = compute_class_weight("balanced", classes=dataframe[target].unique(), y=dataframe[target])
            LOG.info("Using Class weights: %s", class_weights)
        except FileNotFoundError:
            LOG.error("File not found: %s. Exiting.", csv_path)
            raise FileNotFoundError(f"File not found: {csv_path}")
    else:
        class_weights = None
        LOG.info("Not using class weights")

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    LOG.info("Instantiating datamodule <%s>", config.data.datamodule._target_)
    data: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)

    if config.debiasing_method == "adversarial":
        LOG.info("Instantiating adversarial predictor <%s>", AdversarialPredictor)
        module: pl.LightningModule = AdversarialPredictor(**config.predictor, class_weights=class_weights)

    elif config.debiasing_method == "cmmd":
        LOG.info("Instantiating cmmd predictor <%s>", cmmdRegularizedPredictor)
        module: pl.LightningModule = cmmdRegularizedPredictor(**config.predictor, class_weights=class_weights)

    else:
        LOG.info("Instantiating metadata predictor <%s>", MetaDataPrediction)
        module: pl.LightningModule = MetaDataPrediction(**config.predictor, class_weights=class_weights)

    # Init lightning callbacks
    adjust_checkpoint_callback(config)
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for cb_conf in config.callbacks.values():
            if "_target_" in cb_conf:
                LOG.info("Instantiating callback <%s>", cb_conf._target_)
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[pl.LightningLoggerBase] = []
    if "logger" in config:
        for lg_conf in config.logger.values():
            if "_target_" in lg_conf:
                LOG.info("Instantiating logger <%s>", lg_conf._target_)
                l = hydra.utils.instantiate(lg_conf)
                if isinstance(l, WandbLogger):
                    # Log hyperparameters
                    l.log_hyperparams(OmegaConf.to_container(config, resolve=True))

                logger.append(l)

    LOG.info("Instantiating trainer <%s>", config.trainer._target_)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    data.setup("fit")

    LOG.info("Starting training!")
    trainer.fit(module, data)

    wandb.finish()


@hydra.main(config_path="config/debiasing", config_name="adversarial_debiasing.yaml", version_base="1.3")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    main()
