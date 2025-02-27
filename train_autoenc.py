# train autoencoder
import wandb
import logging

import pandas as pd
from pathlib import Path
from typing import List, Optional

import hydra
import yaml
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger

import wandb

LOG = logging.getLogger(__name__)


def train(config: DictConfig):
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    LOG.info("Instantiating datamodule <%s>", config.data.datamodule._target_)
    data: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)

    LOG.info("Instantiating model <%s>", config.autoencoder._target_)
    module: pl.LightningModule = hydra.utils.instantiate(config.autoencoder, _recursive_=True)

    # Init lightning callbacks
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
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger,
    )

    data.setup("fit")

    # Check for a checkpoint path and resume training if it exists
    checkpoint_path: Optional[str] = config.get("checkpoint_path")
    if checkpoint_path:
        LOG.info("Resuming training from checkpoint: %s", checkpoint_path)
    else:
        LOG.info("Starting training from scratch")

    LOG.info("Starting training!")
    trainer.fit(module, data, ckpt_path=checkpoint_path)

    wandb.finish()


@hydra.main(config_path="config", config_name="autoencode.yaml", version_base="1.3")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    main()
