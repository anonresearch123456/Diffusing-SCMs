# train autoencoder
import wandb
import logging
import pandas as pd
from typing import List, Optional

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from src.simulation.debiasing import AdversarialPredictor, MetaDataPrediction, cmmdRegularizedPredictor
from src.simulation.adv import AdversarialMetaDataPrediction

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
        if config.debiasing_method == "adv_orig":
            return
        if config.predictor.target in ["label"]:
            target = config.predictor.target
            monitor = f"val/{target}/bacc"
            config.callbacks.model_checkpoint.monitor = monitor
            config.callbacks.model_checkpoint.mode = "max"
            config.callbacks.model_checkpoint.filename = "epoch-{epoch}-val_bacc-{" + f"{monitor}:.2f" + "}"
            LOG.info("Checkpoint callback adjusted to monitor %s", monitor)


def train(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    LOG.info("Instantiating datamodule <%s>", config.data.datamodule._target_)
    data: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)

    if config.debiasing_method == "adversarial":
        LOG.info("Instantiating adversarial predictor <%s>", AdversarialPredictor)
        module: pl.LightningModule = AdversarialPredictor(**config.predictor)

    elif config.debiasing_method == "cmmd":
        LOG.info("Instantiating cmmd predictor <%s>", cmmdRegularizedPredictor)
        module: pl.LightningModule = cmmdRegularizedPredictor(**config.predictor)

    elif config.debiasing_method == "adv_orig":
        LOG.info("Instantiating adversarial metadata predictor <%s>", AdversarialMetaDataPrediction)
        module: pl.LightningModule = AdversarialMetaDataPrediction(**config.predictor)
    else:
        LOG.info("Instantiating metadata predictor <%s>", MetaDataPrediction)
        module: pl.LightningModule = MetaDataPrediction(**config.predictor)

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


@hydra.main(config_path="config/simulation", config_name="standard.yaml", version_base="1.3")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    main()
