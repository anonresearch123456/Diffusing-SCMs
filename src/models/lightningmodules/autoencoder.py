# aekl_module.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import GradScaler, autocast
import pytorch_lightning as pl
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
# from util import log_reconstructions
from pytorch_lightning.loggers import WandbLogger

import hydra
from omegaconf import DictConfig, OmegaConf


class AEModule(pl.LightningModule):
    def __init__(
        self,
        model: AutoencoderKL,
        discriminator: PatchDiscriminator,
        perceptual_loss: PerceptualLoss,
        adv_loss: PatchAdversarialLoss,
        base_lr: float,
        disc_lr: float,
        adv_weight: float,
        perceptual_weight: float,
        kl_weight: float,
        adv_start: int,
        resample: bool = True,
    ):
        super().__init__()
        self.model = model
        self.discriminator = discriminator
        self.perceptual_loss = perceptual_loss
        self.base_lr = base_lr
        self.disc_lr = disc_lr
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.adv_start = adv_start
        self.resample = resample

        # PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)

        self.adv_loss_fn = adv_loss
        self.scaler_g = GradScaler(device=self.device)
        self.scaler_d = GradScaler(device=self.device)

        self.automatic_optimization = False

        self.save_hyperparameters(ignore=["model", "discriminator", "perceptual_loss", "adv_loss"])

        self.validation_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        optimizer_g, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer_g)

        images = batch["mri"]
        adv_weight = self.adv_weight if self.current_epoch >= self.adv_start else 0.0

        # Generator Step
        self.model.train()
        self.discriminator.eval()

        optimizer_g.zero_grad()

        # hard-coded for now
        with autocast(device_type="cuda"):
            reconstruction, z_mu, z_sigma = self.model(x=images)
            l1_loss = F.l1_loss(reconstruction, images)
            p_loss = self.perceptual_loss(reconstruction, images)
            kl_loss = 0.5 * torch.sum(
                z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4]
            ).mean()

            if adv_weight > 0:
                logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                g_loss = self.adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                g_loss = torch.tensor(0.0, device=self.device)

            loss = (
                l1_loss
                + self.kl_weight * kl_loss
                + self.perceptual_weight * p_loss
                + self.adv_weight * g_loss
            )

            # better safe than sorry
            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = g_loss.mean()


        self.manual_backward(self.scaler_g.scale(loss))
        self.scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.scaler_g.step(optimizer_g)
        self.scaler_g.update()
        self.untoggle_optimizer(optimizer_g)

        # Logging
        self.log_dict({
            "train/loss": loss,
            "train/l1_loss": l1_loss,
            "train/p_loss": p_loss,
            "train/kl_loss": kl_loss,
            "train/g_loss": g_loss,
            "train/lr_g": optimizer_g.param_groups[0]["lr"],
        }, prog_bar=True, logger=True)

        # Discriminator Step
        if adv_weight > 0:
            self.model.eval()
            self.discriminator.train()

            self.toggle_optimizer(optimizer_d)

            optimizer_d.zero_grad()

            # hard-coded for now
            with autocast(device_type="cuda"):
                if self.resample:
                    with torch.no_grad():
                        reconstruction, _, _ = self.model(x=images)
                logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = self.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)

                logits_real = self.discriminator(images.contiguous().detach())[-1]
                loss_d_real = self.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

                d_loss = (loss_d_fake + loss_d_real) * 0.5
                d_loss = d_loss.mean()

            self.manual_backward(self.scaler_d.scale(d_loss))
            self.scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
            self.scaler_d.step(optimizer_d)
            self.scaler_d.update()
            self.untoggle_optimizer(optimizer_d)

        else:
            d_loss = torch.tensor(0.0, device=self.device)

        # Logging
        self.log("train/d_loss", d_loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer_g = Adam(self.model.parameters(), lr=self.base_lr)
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.disc_lr)
        return [optimizer_g, optimizer_d], []

    def validation_step(self, batch, batch_idx):
        images = batch["mri"]
        adv_weight = self.adv_weight if self.current_epoch >= self.adv_start else 0.0

        reconstruction, z_mu, z_sigma = self.model(x=images)
        l1_loss = F.l1_loss(reconstruction, images)
        p_loss = self.perceptual_loss(reconstruction, images)
        kl_loss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4]
        ).mean()

        if adv_weight > 0:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            g_loss = self.adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
        else:
            g_loss = torch.tensor(0.0, device=self.device)

        loss = (
            l1_loss
            + self.kl_weight * kl_loss
            + self.perceptual_weight * p_loss
            + self.adv_weight * g_loss
        )

        loss = loss.mean()
        l1_loss = l1_loss.mean()
        p_loss = p_loss.mean()
        kl_loss = kl_loss.mean()
        g_loss = g_loss.mean()

        if adv_weight > 0:
            reconstruction, _, _ = self.model(x=images)
            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)

            logits_real = self.discriminator(images.contiguous().detach())[-1]
            loss_d_real = self.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

            d_loss = (loss_d_fake + loss_d_real) * 0.5
            d_loss = d_loss.mean()

        else:
            d_loss = torch.tensor(0.0, device=self.device)

        all_losses = {
            "val_loss": loss,
            "val_l1_loss": l1_loss,
            "val_p_loss": p_loss,
            "val_kl_loss": kl_loss,
            "val_g_loss": g_loss,
            "val_d_loss": d_loss,
        }

        self.validation_outputs.append(all_losses)

        # Optionally log reconstructions
        # if batch_idx == 0:
        #     log_reconstructions(images, reconstruction, self.logger, self.global_step)

        return loss

    def on_validation_epoch_end(self):
        # Aggregate metrics
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_outputs]).mean()
        avg_l1_loss = torch.stack([x["val_l1_loss"] for x in self.validation_outputs]).mean()
        avg_p_loss = torch.stack([x["val_p_loss"] for x in self.validation_outputs]).mean()
        avg_kl_loss = torch.stack([x["val_kl_loss"] for x in self.validation_outputs]).mean()
        avg_g_loss = torch.stack([x["val_g_loss"] for x in self.validation_outputs]).mean()
        avg_d_loss = torch.stack([x["val_d_loss"] for x in self.validation_outputs]).mean()

        # Log aggregated metrics
        self.log_dict({
            "val/loss": avg_loss,
            "val/l1_loss": avg_l1_loss,
            "val/p_loss": avg_p_loss,
            "val/kl_loss": avg_kl_loss,
            "val/g_loss": avg_g_loss,
            "val/d_loss": avg_d_loss,
        }, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        # Reset validation outputs
        self.validation_outputs = []

    @classmethod
    def assemble_from_checkpoint(cls, cfg_path: str, checkpoint_path: str):
        cfg = OmegaConf.load(cfg_path)
        model = hydra.utils.instantiate(cfg.autoencoder.model)
        discriminator = hydra.utils.instantiate(cfg.autoencoder.discriminator)
        perceptual_loss = hydra.utils.instantiate(cfg.autoencoder.perceptual_loss)
        adv_loss = hydra.utils.instantiate(cfg.autoencoder.adv_loss)

        model.eval()
        discriminator.eval()
        perceptual_loss.eval()
        adv_loss.eval()

        module = cls.load_from_checkpoint(checkpoint_path, model=model, discriminator=discriminator,
                                            perceptual_loss=perceptual_loss, adv_loss=adv_loss)
        return module

    @classmethod
    def get_encoder_only_from_checkpoint(cls, cfg_path: str, checkpoint_path: str):
        cfg = OmegaConf.load(cfg_path)
        model = hydra.utils.instantiate(cfg.autoencoder.model)


        # Load and filter state_dict correctly for encoder
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        filtered_state_dict = {
            k.replace("model.", ""): v 
            for k, v in checkpoint.items() 
            if k.startswith("model.")
        }

        model.load_state_dict(filtered_state_dict)
        model.eval()

        return model
