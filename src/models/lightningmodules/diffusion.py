# ldm_module.py
import hydra
from src.models.lightningmodules.autoencoder import AEModule
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL


class TabularEncoder(torch.nn.Module):
    """ Simple MLP for encoding tabular data """
    def __init__(self, input_dim, hidden_dim, output_dim, non_linear = False):
        super().__init__()
        if non_linear:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Linear(hidden_dim, output_dim)
            )
        
    def forward(self, x):
        return self.encoder(x)


class LDMModule(pl.LightningModule):
    def __init__(
        self,
        diffusion: DiffusionModelUNet,
        autoencoder: AutoencoderKL,
        scheduler: DDPMScheduler,
        tabular_encoder: TabularEncoder,
        base_lr: float,
        scale_factor: float = 1.0,
        padding: bool = False,
        concat_cond: bool = True,
        cross_cond: bool = True,
    ):
        super().__init__()
        self.model = diffusion
        self.autoencoder = autoencoder
        self.scheduler = scheduler
        self.base_lr = base_lr
        self.scale_factor = scale_factor
        self.tabular_encoder = tabular_encoder
        self.padding = padding
        self.concat_cond = concat_cond
        self.cross_cond = cross_cond

        self.scaler = GradScaler()

        self.automatic_optimization = False

        self.save_hyperparameters(ignore=['model', 'autoencoder', 'scheduler', 'tabular_encoder'])
        self.validation_outputs = []

        # Freeze the autoencoder
        self.freeze_module(self.autoencoder)

    def freeze_module(self, module):
        """Freeze all parameters in a given module."""
        module.eval()  # Sets the module to evaluation mode
        for param in module.parameters():
            param.requires_grad = False

    def state_dict(self):
        # save everything except the encoder
        return {k: v for k, v in super().state_dict().items() if "autoencoder" not in k}

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to exclude autoencoder's parameters.
        This prevents PyTorch Lightning from expecting autoencoder parameters in the checkpoint.
        """
        autoencoder_prefix = 'autoencoder.'
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(autoencoder_prefix)}
        return super().load_state_dict(filtered_state_dict, strict=False)

    def forward(self, x, timesteps, context):
        x = F.pad(x, (1, 1, 2, 2, 2, 2), mode='reflect') if self.padding else x
        out = self.model(x, timesteps=timesteps, context=context)
        # remove padding
        out = out[:, :, 2:-2, 2:-2, 1:-1] if self.padding else out
        return out

    def on_train_start(self) -> None:
        self.autoencoder.eval()

    def training_step(self, batch, batch_idx):
        images: torch.Tensor = batch['mri']
        tabular_context: torch.Tensor = batch['context']

        batch_size = images.size(0)
        device = self.device

        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=device).long()

        optimizer = self.optimizers()
        optimizer.zero_grad()

        with autocast(str(device)):
            # Encode images using AE
            with torch.no_grad():
                encoded_image, _ = self.autoencoder.encode(images)
                encoded_image = encoded_image * self.scale_factor

            # Get tab embeddings
            context_embeddings = self.tabular_encoder(tabular_context) if self.cross_cond else None

            # Generate noise
            noise = torch.randn_like(encoded_image).to(device)
            noisy_e = self.scheduler.add_noise(original_samples=encoded_image, noise=noise, timesteps=timesteps)

            # append tabular_context as to MRI channels, broadcasted;
            # note tabular context has shape (batch_size, seq_len, tab_dim)
            tabular_context_v = tabular_context.view(tabular_context.size(0), tabular_context.size(2), 1, 1, 1)
            tabular_context_v = tabular_context_v.expand(-1, -1, encoded_image.size(2), encoded_image.size(3), encoded_image.size(4))
            noisy_e = torch.cat([noisy_e, tabular_context_v], dim=1) if self.concat_cond else noisy_e

            # Predict noise
            noise_pred: torch.Tensor = self.forward(noisy_e, timesteps, context_embeddings)

            if self.scheduler.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(encoded_image, noise, timesteps)
            elif self.scheduler.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unknown prediction type {self.scheduler.prediction_type}")

            loss = F.mse_loss(noise_pred.float(), target.float())

        self.manual_backward(self.scaler.scale(loss))
        self.scaler.step(optimizer)
        self.scaler.update()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images: torch.Tensor = batch['mri']
        tabular_context: torch.Tensor = batch['context']

        batch_size = images.size(0)
        device = images.device

        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=device).long()

        # Encode images using AE
        with torch.no_grad():
            encoded_image, _ = self.autoencoder.encode(images)
            encoded_image = encoded_image * self.scale_factor

        # Get tab embeddings
        context_embeddings = self.tabular_encoder(tabular_context) if self.cross_cond else None

        # Generate noise
        noise = torch.randn_like(encoded_image).to(device)
        noisy_e = self.scheduler.add_noise(original_samples=encoded_image, noise=noise, timesteps=timesteps)

        # append tabular_context as to MRI channels, broadcasted;
        # note tabular context has shape (batch_size, seq_len, tab_dim)
        tabular_context_v = tabular_context.view(tabular_context.size(0), tabular_context.size(2), 1, 1, 1)
        tabular_context_v = tabular_context_v.expand(-1, -1, encoded_image.size(2), encoded_image.size(3), encoded_image.size(4))
        noisy_e = torch.cat([noisy_e, tabular_context_v], dim=1) if self.concat_cond else noisy_e

        # Predict noise
        noise_pred: torch.Tensor = self.forward(noisy_e, timesteps, context_embeddings)

        if self.scheduler.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(encoded_image, noise, timesteps)
        elif self.scheduler.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.prediction_type}")

        loss = F.mse_loss(noise_pred.float(), target.float())

        self.validation_outputs.append(loss.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_outputs).mean()
        self.log("val/loss", avg_loss, prog_bar=True)
        self.validation_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(
            list(self.model.parameters()) + list(self.tabular_encoder.parameters()),
            lr=self.base_lr
        )
        return optimizer

    @classmethod
    def get_diffusion_for_inference(cls, cfg: str | DictConfig, checkpoint_path: str) -> tuple[AutoencoderKL, DiffusionModelUNet, DDIMScheduler, TabularEncoder]:
        config = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

        # Load and filter state_dict correctly for encoder
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        encoder: AutoencoderKL = AEModule.get_encoder_only_from_checkpoint(cfg_path=config.autoencoder.cfg_path,
                                                                           checkpoint_path=config.autoencoder.checkpoint_path)
        encoder.eval()

        # for inference we use DDIM scheduler instead of DDPMScheduler
        requird_scheduler_keys = ['num_train_timesteps', 'prediction_type', 'beta_start', 'beta_end', 'schedule']
        scheduler: DDIMScheduler = DDIMScheduler(**{k: config.ldm.scheduler[k] for k in requird_scheduler_keys})

        # Load tabular encoder
        tabular_encoder: TabularEncoder = hydra.utils.instantiate(config.ldm.tabular_encoder)
        tabular_encoder_state_dict = {k.replace('tabular_encoder.', ''): v for k, v in checkpoint.items() if 'tabular_encoder' in k}
        tabular_encoder.load_state_dict(tabular_encoder_state_dict)
        tabular_encoder.eval()

        # load diffusion model
        diffusion_model: DiffusionModelUNet = hydra.utils.instantiate(config.ldm.diffusion)
        diffusion_model_state_dict = {k.replace('model.', ''): v for k, v in checkpoint.items() if 'model' in k}
        diffusion_model.load_state_dict(diffusion_model_state_dict)
        diffusion_model.eval()

        return encoder, diffusion_model, scheduler, tabular_encoder
