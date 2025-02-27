# aekl_module.py
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
# from util import log_reconstructions
from pytorch_lightning.loggers import WandbLogger
from src.models.networks.resnet import ThreeDResNet, ThreeDResNetFixedChannels, ThreeDResNetEncoded
from src.models.networks.resnet_gn import ThreeDResNetGN

import hydra
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score, average_precision_score


# This module is used to predict the metadata from the MRI images
# It uses the feature layer output from the resnet encoder to predict the metadata
class PredictionHead(torch.nn.Module):
    def __init__(self, in_features: int, n_outputs: int):
        super().__init__()
        # use layers with ReLU activation
        self.fc1 = torch.nn.Linear(in_features, 2*in_features)
        self.fc2 = torch.nn.Linear(2*in_features, in_features)
        self.fc3 = torch.nn.Linear(in_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MetaDataPredictionAbstract(pl.LightningModule):
    def __init__(
        self,
        resnet_cfg: dict,
        lr_start: float,
        lr_end: float,
        predictor_type: str,
        target: str,
        encoder: Optional[AutoencoderKL] = None,
        class_weights: Optional[np.ndarray | torch.Tensor] = None,
        encoder_predictor: str = "standard",
        predictor_backbone: str = "resnet",
    ):
        super().__init__()

        # predictor_type: "full" or "encoded"
        # full: can be reconstructed or original input
        if encoder is None and predictor_type == "encoded":
            raise ValueError("If predictor_type is 'encoded', encoder must be provided.")

        if predictor_type not in ["reconstruct", "encoded", "pristine"]:
            raise ValueError(f"predictor_type must be 'reconstruct', 'pristine' or 'encoded', but got {predictor_type}")

        if encoder:
            self.freeze_module(encoder)

        # adjust the number of outputs based on the target
        resnet_cfg["n_outputs"] = 1 if target in ["age","sex","age_std","all"] else 3

        if not encoder:
            if predictor_backbone == "resnet":
                self.predictor = ThreeDResNet(**resnet_cfg)
            elif predictor_backbone == "resnet_gn":
                self.predictor = ThreeDResNetGN(**resnet_cfg)
            else:
                raise ValueError(f"Unsupported predictor backbone: {predictor_backbone}")
        else:
            self.predictor = ThreeDResNetFixedChannels(**resnet_cfg) if encoder_predictor == "fixed" else ThreeDResNetEncoded(**resnet_cfg)

        self.encoder = encoder
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.target = target

        self.automatic_optimization = False

        self.predictor_type = predictor_type

        self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.scaler = GradScaler(device=self.device)


        self.save_hyperparameters(ignore=["predictor", "encoder"])

        self.validation_outputs: List[Tuple[torch.Tensor]] | Dict[str, List[Tuple[torch.Tensor]]] = []
        self.test_outputs: List[Tuple[torch.Tensor]] | Dict[str, List[Tuple[torch.Tensor]]] = []

    def state_dict(self):
        # save everything except the encoder
        return {k: v for k, v in super().state_dict().items() if "encoder" not in k}

    def load_state_dict(self, state_dict, strict=True):
        # load everything except the encoder
        encoder_prefix = "encoder."
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(encoder_prefix)}
        return super().load_state_dict(filtered_state_dict, strict=False)

    def freeze_module(self, module):
        """Freeze all parameters in a given module."""
        module.eval()  # Sets the module to evaluation mode
        for param in module.parameters():
            param.requires_grad = False

    def get_loss_function(self, target: str, weights: Optional[torch.Tensor] = None):
        if target in ["age", "age_std"]:
            return F.mse_loss
        elif target == "sex":
            return BCEWithLogitsLoss(weight=weights)
        elif target == "site":
            return CrossEntropyLoss(weight=weights)
        else:
            raise ValueError(f"Unsupported target type: {target}")

    def forward(self, x) -> torch.Tensor:
        if self.predictor_type == "reconstruct":
            with torch.no_grad():
                reconstruct = self.encoder.reconstruct(x)
            return self.predictor(reconstruct)[0]
            # no encoder version
        elif self.predictor_type == "pristine":
            return self.predictor(x)[0]
        elif self.predictor_type == "encoded":
            with torch.no_grad():
                encoded, _ = self.encoder.encode(x)
            return self.predictor(encoded)[0]

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def calculate_log_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, target: str):
        if target in ["age", "age_std"]:
            mae = F.l1_loss(y_hat, y)
            mse = F.mse_loss(y_hat, y)
            r2 = r2_score(y.numpy(), y_hat.numpy())
            self.log("val/mae", mae)
            self.log("val/mse", mse)
            self.log("val/r2", r2)

            loss = mse
        # report balanced accuracy for sex
        elif target == "sex":
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            self.log("val/bacc", bacc)

            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # report balanced accuracy for site
        elif target == "site":
            y_pred = y_hat.argmax(dim=1)
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            self.log("val/bacc", bacc)

            loss = F.cross_entropy(y_hat, y)
        else:
            raise ValueError(f"Unsupported target type: {target}")

        # for correct checkpointing
        self.log("val/loss", loss)

    def calculate_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, target: str) -> dict[torch.Tensor]:
        if target in ["age", "age_std"]:
            mae = F.l1_loss(y_hat, y)
            mse = F.mse_loss(y_hat, y)
            r2 = r2_score(y.numpy(), y_hat.numpy())
            pearson_corr = torch.tensor(np.corrcoef(y.numpy(), y_hat.numpy())[0, 1])
            return {"mae": mae.item(), "mse": mse.item(), "r2": r2, "pearson_corr": pearson_corr.item()}
        # report balanced for sex
        elif target == "sex":
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            roc_auc = roc_auc_score(y.numpy(), torch.sigmoid(y_hat).numpy())
            return {"bacc": bacc, "roc_auc": roc_auc}
        # report balanced accuracy for site
        elif target == "site":
            y_pred = y_hat.argmax(dim=1)
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            roc_auc = roc_auc_score(y.numpy(), F.softmax(y_hat, dim=1).numpy(), multi_class="ovo")
            return {"bacc": bacc, "roc_auc": roc_auc}
        else:
            raise ValueError(f"Unsupported target type: {target}")

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass


class MetaDataPrediction(MetaDataPredictionAbstract):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.loss = self.get_loss_function(self.target, self.class_weights)

        self.save_hyperparameters(ignore=["predictor", "encoder"])

        self.validation_outputs: List[Tuple[torch.Tensor]] = []
        self.test_outputs: List[Tuple[torch.Tensor]] = []

    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        optimizer.zero_grad()

        x: torch.Tensor = batch["mri"]
        y: torch.Tensor = batch[self.target]

        y_hat: torch.Tensor = self.forward(x).squeeze()
        loss = self.loss(y_hat.squeeze(), y)

        self.manual_backward(loss)
        optimizer.step()

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["mri"]
        y: torch.Tensor = batch[self.target]

        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat.squeeze(), y)

        self.validation_outputs.append((y.detach().cpu(), y_hat.detach().cpu()))
        return loss
    
    def test_step(self, batch, batch_idx):
        x: torch.Tensor = batch["mri"]
        y: torch.Tensor = batch[self.target]

        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat.squeeze(), y)

        self.test_outputs.append((y.detach().cpu(), y_hat.detach().cpu()))
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.predictor.parameters(), lr=self.lr_start)
        return optimizer

    def on_validation_epoch_end(self):
        # aggregate predictions
        y = torch.cat([y for y, _ in self.validation_outputs])
        y_hat = torch.cat([y_hat for _, y_hat in self.validation_outputs])

        self.calculate_log_metrics(y, y_hat, self.target)

        self.validation_outputs = []

    def on_test_epoch_end(self):
        # aggregate predictions
        y = torch.cat([y for y, _ in self.test_outputs])
        y_hat = torch.cat([y_hat for _, y_hat in self.test_outputs])

        return self.calculate_metrics(y, y_hat, self.target)


class MetadataPredictionAll(MetaDataPredictionAbstract):
    def __init__(
        self,
        lambda_age: float,
        lambda_sex: float,
        lambda_site: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.loss_age = self.get_loss_function("age_std")
        self.loss_sex = self.get_loss_function("sex")
        self.loss_site = self.get_loss_function("site", weights=self.class_weights)

        self.lambda_age = lambda_age
        self.lambda_sex = lambda_sex
        self.lambda_site = lambda_site

        self.save_hyperparameters(ignore=["predictor", "encoder"])

        self.validation_outputs: Dict[str, List[Tuple[torch.Tensor]]] = {
            "age_std": [],
            "sex": [],
            "site": []
        }
        self.test_outputs: Dict[str, List[Tuple[torch.Tensor]]] = {
            "age_std": [],
            "sex": [],
            "site": []
        }

        self.age_pred_head = PredictionHead(self.predictor.feature_size, 1)
        self.sex_pred_head = PredictionHead(self.predictor.feature_size, 1)
        self.site_pred_head = PredictionHead(self.predictor.feature_size, 3)

    def configure_optimizers(self):
        optimizer = AdamW(
            list(self.predictor.parameters()) + list(self.age_pred_head.parameters()) + list(self.sex_pred_head.parameters()) + list(self.site_pred_head.parameters()),
            lr=self.lr_start
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr_end)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x) -> torch.Tensor:
        if self.predictor_type == "reconstruct":
            with torch.no_grad():
                reconstruct = self.encoder.reconstruct(x)
            return self.predictor(reconstruct)[1]
            # no encoder version
        elif self.predictor_type == "pristine":
            return self.predictor(x)[1]
        elif self.predictor_type == "encoded":
            with torch.no_grad():
                encoded, _ = self.encoder.encode(x)
            return self.predictor(encoded)[1]

    def get_predictions(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.forward(x)

        age_hat = self.age_pred_head(features).squeeze()
        sex_hat = self.sex_pred_head(features).squeeze()
        site_hat = self.site_pred_head(features).squeeze()

        return age_hat, sex_hat, site_hat

    def get_prediction_losses(self, x, age_true, sex_true, site_true) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        age_hat, sex_hat, site_hat = self.get_predictions(x)

        loss_age = self.loss_age(age_hat, age_true)
        loss_sex = self.loss_sex(sex_hat, sex_true)
        loss_site = self.loss_site(site_hat, site_true)

        return loss_age, loss_sex, loss_site

    def get_prediction_losses_no_reduction(self, x, age_true, sex_true, site_true) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        age_hat, sex_hat, site_hat = self.get_predictions(x)

        loss_age = F.mse_loss(age_hat, age_true, reduction="none")
        loss_sex = F.binary_cross_entropy_with_logits(sex_hat, sex_true, reduction="none")
        loss_site = F.cross_entropy(site_hat, site_true, reduction="none")

        return loss_age, loss_sex, loss_site

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        x: torch.Tensor = batch["mri"]
        age: torch.Tensor = batch["age_std"]
        sex: torch.Tensor = batch["sex"]
        site: torch.Tensor = batch["site"]

        with (torch.amp.autocast(device_type=str(self.device)) if self.predictor_type != "pristine" else nullcontext()):
            features = self.forward(x)

            age_hat = self.age_pred_head(features).squeeze()
            sex_hat = self.sex_pred_head(features).squeeze()
            site_hat = self.site_pred_head(features).squeeze()

            loss_age = self.loss_age(age_hat, age)
            loss_sex = self.loss_sex(sex_hat, sex)
            loss_site = self.loss_site(site_hat, site)

            self.log("train/age_loss", loss_age)
            self.log("train/sex_loss", loss_sex)
            self.log("train/site_loss", loss_site)

            loss = self.lambda_age * loss_age + self.lambda_sex * loss_sex + self.lambda_site * loss_site

        if self.predictor_type != "pristine":
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            self.manual_backward(loss)
            optimizer.step()

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["mri"]
        age: torch.Tensor = batch["age_std"]
        site: torch.Tensor = batch["site"]
        sex: torch.Tensor = batch["sex"]

        with (torch.amp.autocast(device_type=str(self.device)) if self.predictor_type != "pristine" else nullcontext()): 
            features = self.forward(x)

            age_hat = self.age_pred_head(features).squeeze()
            sex_hat = self.sex_pred_head(features).squeeze()
            site_hat = self.site_pred_head(features).squeeze()

            loss_age = self.loss_age(age_hat, age)
            loss_sex = self.loss_sex(sex_hat, sex)
            loss_site = self.loss_site(site_hat, site)

            loss = self.lambda_age * loss_age + self.lambda_sex * loss_sex + self.lambda_site * loss_site

        self.validation_outputs["age_std"].append((age.detach().cpu(), age_hat.detach().cpu()))
        self.validation_outputs["sex"].append((sex.detach().cpu(), sex_hat.detach().cpu()))
        self.validation_outputs["site"].append((site.detach().cpu(), site_hat.detach().cpu()))

        return loss

    def test_step(self, batch, batch_idx):
        x: torch.Tensor = batch["mri"]
        age: torch.Tensor = batch["age_std"]
        site: torch.Tensor = batch["site"]
        sex: torch.Tensor = batch["sex"]

        with (torch.amp.autocast(device_type=str(self.device)) if self.predictor_type != "pristine" else nullcontext()): 
            features = self.forward(x)

            age_hat = self.age_pred_head(features).squeeze()
            sex_hat = self.sex_pred_head(features).squeeze()
            site_hat = self.site_pred_head(features).squeeze()

            loss_age = self.loss_age(age_hat, age)
            loss_sex = self.loss_sex(sex_hat, sex)
            loss_site = self.loss_site(site_hat, site)

            loss = self.lambda_age * loss_age + self.lambda_sex * loss_sex + self.lambda_site * loss_site

        self.test_outputs["age_std"].append((age.detach().cpu(), age_hat.detach().cpu()))
        self.test_outputs["sex"].append((sex.detach().cpu(), sex_hat.detach().cpu()))
        self.test_outputs["site"].append((site.detach().cpu(), site_hat.detach().cpu()))

        return loss

    def on_validation_epoch_end(self):
        loss = 0.0
        combined_metric = 0.0
        get_lambda = lambda x: self.lambda_age if x in ["age", "age_std"] else self.lambda_sex if x == "sex" else self.lambda_site
        for target in self.validation_outputs.keys():
            y = torch.cat([y for y, _ in self.validation_outputs[target]])
            y_hat = torch.cat([y_hat for _, y_hat in self.validation_outputs[target]])

            l, m = self.calculate_log_metrics(y, y_hat, target)
            loss += get_lambda(target) * l
            combined_metric += m
        
        # especially for checkpointing
        self.log("val/loss", loss)
        self.log("val/combined_metric", combined_metric / 3.0)
        self.validation_outputs = {k: [] for k in self.validation_outputs.keys()}

        # do scheduler step if it is not a sanity check
        if not self.trainer.sanity_checking:
            self.lr_schedulers().step()

    def on_test_epoch_end(self):
        out = {}
        for target in self.test_outputs.keys():
            y = torch.cat([y for y, _ in self.test_outputs[target]])
            y_hat = torch.cat([y_hat for _, y_hat in self.test_outputs[target]])
            out.update(self.calculate_metrics(y, y_hat, target))
        self.test_outputs = {k: [] for k in self.test_outputs.keys()}
        return out

    def calculate_log_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, target: str):
        if target in ["age", "age_std"]:
            mae = F.l1_loss(y_hat, y)
            mse = F.mse_loss(y_hat, y)
            r2 = r2_score(y.numpy(), y_hat.numpy())
            self.log("val/age_std/mae", mae)
            self.log("val/age_std/mse", mse)
            self.log("val/age_std/r2", r2)

            loss = mse
            metric = r2
        # report balanced accuracy for sex
        elif target == "sex":
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            self.log("val/sex/bacc", bacc)

            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            metric = bacc
        # report balanced accuracy for site
        elif target == "site":
            y_pred = y_hat.argmax(dim=1)
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            self.log("val/site/bacc", bacc)

            loss = F.cross_entropy(y_hat, y)
            metric = bacc
        else:
            raise ValueError(f"Unsupported target type: {target}")

        # for correct checkpointing
        return loss, metric

    def calculate_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, target: str) -> dict[torch.Tensor]:
        if target in ["age", "age_std"]:
            mae = F.l1_loss(y_hat, y)
            mse = F.mse_loss(y_hat, y)
            r2 = r2_score(y.numpy(), y_hat.numpy())
            pearson_corr = torch.tensor(np.corrcoef(y.numpy(), y_hat.numpy())[0, 1])
            return {"age_std/mae": mae.item(), "age_std/mse": mse.item(), "age_std/r2": r2, "age_std/pearson_corr": pearson_corr.item()}
        # report balanced for sex
        elif target == "sex":
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            roc_auc = roc_auc_score(y.numpy(), torch.sigmoid(y_hat).numpy())
            pr_auc = average_precision_score(y.numpy(), torch.sigmoid(y_hat).numpy())
            return {"sex/bacc": bacc, "sex/roc_auc": roc_auc, "sex/pr_auc": pr_auc}
        # report balanced accuracy for site
        elif target == "site":
            y_pred = y_hat.argmax(dim=1)
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            roc_auc = roc_auc_score(y.numpy(), F.softmax(y_hat, dim=1).numpy(), multi_class="ovo")
            pr_auc = average_precision_score(y.numpy(), F.softmax(y_hat, dim=1).numpy(), average='macro')
            return {"site/bacc": bacc, "site/roc_auc": roc_auc, "site/pr_auc": pr_auc}
        else:
            raise ValueError(f"Unsupported target type: {target}")
