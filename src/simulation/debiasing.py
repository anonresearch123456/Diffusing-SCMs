# aekl_module.py
import numpy as np
from src.simulation.resnet import TwoDResNet, ConvNet
from src.simulation.resnet_gn import TwoDResNetGN
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from pytorch_lightning.loggers import WandbLogger
from src.debiasing.cmmd_utils import mmd_compute

import hydra
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score


class CorrelationLoss(torch.nn.Module):
    def forward(self, inp: torch.Tensor, target: torch.Tensor):
        in_mean = inp.mean()
        tar_mean = target.mean()

        in_centered = inp - in_mean
        tar_centered = target - tar_mean  # TODO wird 0, wenn nur eine Klasse des Targets in Batch -> nan im backward

        r_numerator = torch.sum(in_centered * tar_centered)
        r_denominator = torch.sqrt((torch.sum(in_centered**2)) * torch.sum(tar_centered**2)) + 1e-5  # TODO ziemlich "groß". 1e-7?

        r = r_numerator / r_denominator

        r = torch.clamp(r, min=-1.0, max=1.0)
        return r**2


class CorrelationLossNegative(torch.nn.Module):
    def forward(self, inp: torch.Tensor, target: torch.Tensor):
        in_mean = inp.mean()
        tar_mean = target.mean()

        in_centered = inp - in_mean
        tar_centered = target - tar_mean

        r_numerator = torch.sum(in_centered * tar_centered)
        r_denominator = torch.sqrt((torch.sum(in_centered**2)) * torch.sum(tar_centered**2)) + 1e-5

        r = r_numerator / r_denominator

        r = torch.clamp(r, min=-1.0, max=1.0)

        return -r**2


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


class PredictionHeadLinear(torch.nn.Module):
    def __init__(self, in_features: int, n_outputs: int):
        super().__init__()
        # use layers with ReLU activation
        self.fc = torch.nn.Linear(in_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MetaDataPredictionAbstract(pl.LightningModule):
    def __init__(
        self,
        resnet_cfg: dict,
        lr_start: float,
        lr_end: float,
        target: str,
        class_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        encoder_type: str = "resnet",
        pred_head: str = "linear",
    ):
        super().__init__()
        # For our new pipeline the target is always "label" (binary)
        # and we force n_outputs=1.
        resnet_cfg["n_outputs"] = 1  
        # Use the new 2D ResNet
        if encoder_type == "resnet":
            self.predictor = TwoDResNet(**resnet_cfg)
        elif encoder_type == "resnet_gn":
            self.predictor = TwoDResNetGN(**resnet_cfg)
        elif encoder_type == "convnet":
            self.predictor = ConvNet(**resnet_cfg)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        # Choose between a linear head or a non-linear one.
        prediction_head = PredictionHeadLinear if pred_head == "linear" else PredictionHead

        self.classifier_task = prediction_head(
            self.predictor.feature_size, 
            1  # n_outputs is 1 (for binary prediction)
        )

        self.lr_start = lr_start
        self.lr_end = lr_end
        self.target = target  # should be "label" in our case

        # We use manual optimization.
        self.automatic_optimization = False

        self.class_weights = (torch.as_tensor(class_weights, dtype=torch.float32)
                              if class_weights is not None else None)

        self.save_hyperparameters(ignore=["predictor"])

        self.validation_outputs: Dict[str, List[Tuple[torch.Tensor]]] = {
            "label": [],
            "cf_std": []
        }
        self.test_outputs: Dict[str, List[Tuple[torch.Tensor]]] = {
            "label": [],
            "cf_std": []
        }

    def get_loss_function(self, target: str, weights: Optional[torch.Tensor] = None):
        # Here we assume that "label" is binary.
        if target in ["label"]:
            return BCEWithLogitsLoss(weight=weights)
        elif target in ["cf", "cf_std"]:
            return F.mse_loss
        else:
            raise ValueError(f"Unsupported target type: {target}")

    def forward(self, x) -> torch.Tensor:
        # Return classifier output (using features from the predictor)
        return self.classifier_task(self.predictor(x)[1])

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def calculate_log_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, target: str):
        # For binary classification we log balanced accuracy and binary cross entropy.
        if target == "label":
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
            self.log("val/label/bacc", bacc)
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        else:
            raise ValueError(f"Unsupported target type: {target}")
        self.log("val/loss", loss)
        return loss

    def calculate_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, target: str) -> dict:
        if target == "label":
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = roc_auc_score(y.cpu().numpy(), torch.sigmoid(y_hat).cpu().numpy())
            return {"label/bacc": bacc, "label/roc_auc": roc_auc}
        else:
            raise ValueError(f"Unsupported target type: {target}")

    def on_validation_epoch_end(self):
        raise NotImplementedError

    def on_test_epoch_end(self):
        raise NotImplementedError


class MetaDataPrediction(MetaDataPredictionAbstract):
    def __init__(
        self,
        pred_head: str = "linear",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss = self.get_loss_function(self.target, self.class_weights)
        self.pred_head = pred_head

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.classifier_task(self.predictor(x)[1])

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        x: torch.Tensor = batch["img"]  # our new data uses key "img"
        y: torch.Tensor = batch[self.target]  # target is "label"

        y_hat: torch.Tensor = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        self.manual_backward(loss)
        optimizer.step()

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["img"]
        y: torch.Tensor = batch[self.target]

        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        self.validation_outputs.append((y.detach().cpu(), y_hat.detach().cpu()))
        return loss
    
    def test_step(self, batch, batch_idx):
        x: torch.Tensor = batch["img"]
        y: torch.Tensor = batch[self.target]

        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        self.test_outputs.append((y.detach().cpu(), y_hat.detach().cpu()))
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(list(self.predictor.parameters()) + list(self.classifier_task.parameters()), lr=self.lr_start)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr_end)
        return [optimizer], [scheduler]

    def on_validation_epoch_end(self):
        y = torch.cat([y for y, _ in self.validation_outputs])
        y_hat = torch.cat([y_hat for _, y_hat in self.validation_outputs])

        self.calculate_log_metrics(y, y_hat, self.target)
        self.validation_outputs = []
        if not self.trainer.sanity_checking:
            self.lr_schedulers().step()

    def on_test_epoch_end(self):
        y = torch.cat([y for y, _ in self.test_outputs])
        y_hat = torch.cat([y_hat for _, y_hat in self.test_outputs])
        metrics = self.calculate_metrics(y, y_hat, self.target)
        self.test_outputs = []
        return metrics


class AdversarialPredictor(MetaDataPredictionAbstract):
    def __init__(
        self,
        protected_attributes: list,  # e.g. ["cf"]
        protected_pred_head: str = "linear",
        lambda_protected: float = 1.0,
        lambda_feature: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss = self.get_loss_function(self.target, self.class_weights)
        self.lambda_protected = lambda_protected
        self.lambda_feature = lambda_feature
        self.protected_attributes = protected_attributes

        # Choose prediction head for protected attributes.
        prediction_head = PredictionHeadLinear if protected_pred_head == "linear" else PredictionHead

        self.protected_losses = {k: self.get_loss_function(k) for k in protected_attributes}
        self.feature_losses = {k: CorrelationLossNegative() for k in protected_attributes}
        
        self.pred_heads_protected = torch.nn.ModuleDict(
            {
                k: prediction_head(
                    self.predictor.feature_size, 
                    1  # one output for continuous protected attribute "cf"
                )
                for k in protected_attributes
            }
        )

        self.classifier_task = prediction_head(
            self.predictor.feature_size, 
            1  # target "label"
        )

        self.save_hyperparameters(ignore=["predictor", "pred_heads"])

    def forward(self, x):
        return self.classifier_task(self.predictor(x)[1])

    def configure_optimizers(self):
        optimizer_full = Adam(list(self.predictor.parameters()) + list(self.classifier_task.parameters()), lr=self.lr_start)
        optimizer_protected = [Adam(p.parameters(), lr=self.lr_start) for p in self.pred_heads_protected.values()]
        optimizer_features = Adam(self.predictor.parameters(), lr=self.lr_start)

        scheduler_full = CosineAnnealingLR(optimizer_full, T_max=self.trainer.max_epochs, eta_min=self.lr_end)
        scheduler_protected = [CosineAnnealingLR(o, T_max=self.trainer.max_epochs, eta_min=self.lr_end) for o in optimizer_protected]
        scheduler_features = CosineAnnealingLR(optimizer_features, T_max=self.trainer.max_epochs, eta_min=self.lr_end)

        return [optimizer_full, optimizer_features, *optimizer_protected], [scheduler_full, scheduler_features, *scheduler_protected]

    def split_batch(self, batch) -> list:
        """
        Splits the batch based on the target "label".
        Returns two dictionaries: one for label == 0 and one for label == 1.
        """
        if self.target == "label":
            mask = batch["label"] == 0
            return [
                {k: v[mask] for k, v in batch.items()},
                {k: v[~mask] for k, v in batch.items()}
            ]
        else:
            raise ValueError(f"Unsupported target for splitting: {self.target}")

    def training_step(self, batch, batch_idx):

        optimizer_full, optimizer_features, *optimizer_protected = self.optimizers()
        optimizer_full.zero_grad()
        optimizer_features.zero_grad()
        for o in optimizer_protected:
            o.zero_grad()

        x: torch.Tensor = batch["img"]
        y: torch.Tensor = batch[self.target]

        y_hat: torch.Tensor = self.forward(x).squeeze()
        loss = self.loss(y_hat.squeeze(), y)

        self.manual_backward(loss)
        optimizer_full.step()
        optimizer_full.zero_grad()

        self.log("train/loss", loss)

        with torch.no_grad():
            features = self.predictor(x)[1]

        for protected_attribute, o in zip(self.protected_attributes, optimizer_protected):
            y_protected = batch[protected_attribute]
            y_hat_protected = self.pred_heads_protected[protected_attribute](features).squeeze()
            loss_protected = self.protected_losses[protected_attribute](y_hat_protected, y_protected) * self.lambda_protected
            self.manual_backward(loss_protected)
            self.log(f"train/loss_{protected_attribute}", loss_protected.detach().cpu())
            o.step()
            o.zero_grad()

        batches = self.split_batch(batch)
        for b in batches:
            if not b or len(b["img"]) == 0:
                continue
            x = b["img"]
            features = self.predictor(x)[1]
            total_loss = 0.0
            for protected_attribute in self.protected_attributes:
                y_protected = b[protected_attribute].squeeze()
                y_hat_protected = self.pred_heads_protected[protected_attribute](features).squeeze()
                loss_protected = - self.feature_losses[protected_attribute](y_hat_protected, y_protected) * self.lambda_feature
                total_loss += loss_protected
            total_loss = total_loss / (len(self.protected_attributes) * 2)
            self.manual_backward(total_loss)
            optimizer_features.step()
            optimizer_features.zero_grad()
        return loss

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["img"]
        y: torch.Tensor = batch[self.target]
        features = self.predictor(x)[1]
        y_hat = self.classifier_task(features).squeeze()
        loss = self.loss(y_hat, y)

        self.validation_outputs[self.target].append((y.detach().cpu(), y_hat.detach().cpu()))
        batches = self.split_batch(batch)

        for i, b in enumerate(batches):
            features_b = self.predictor(b["img"])[1]
            b["features"] = features_b

        for i, b in enumerate(batches):
            for protected_attribute in self.protected_attributes:
                y_protected = b[protected_attribute]
                y_hat_protected = self.pred_heads_protected[protected_attribute](b["features"]).squeeze()
                loss_protected = self.protected_losses[protected_attribute](y_hat_protected, y_protected)
                self.validation_outputs[protected_attribute].append((y_protected.detach().cpu(), y_hat_protected.detach().cpu()))
                self.log(f"val/loss_{protected_attribute}_{i}", loss_protected.detach().cpu())
        return loss

    def test_step(self, batch, batch_idx):
        x: torch.Tensor = batch["img"]
        y: torch.Tensor = batch[self.target]
        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        self.test_outputs[self.target].append((y.detach().cpu(), y_hat.detach().cpu()))

        for protected_attribute in self.protected_attributes:
            y_protected = batch[protected_attribute]
            y_hat_protected = self.pred_heads_protected[protected_attribute](self.predictor(x)[1]).squeeze()
            loss_protected = self.protected_losses[protected_attribute](y_hat_protected, y_protected)
            self.test_outputs[protected_attribute].append((y_protected.detach().cpu(), y_hat_protected.detach().cpu()))
        return loss

    def on_validation_epoch_end(self):
        loss = 0.0
        for target in self.validation_outputs.keys():
            if len(self.validation_outputs[target]) == 0:
                continue
            y = torch.cat([y for y, _ in self.validation_outputs[target]])
            y_hat = torch.cat([y_hat for _, y_hat in self.validation_outputs[target]])
            l, m = self.calculate_log_metrics(y, y_hat, target)
            if target == self.target:
                loss = l
        self.log("val/loss", loss)
        self.validation_outputs = {k: [] for k in self.validation_outputs.keys()}
        if not self.trainer.sanity_checking:
            for scheduler in self.lr_schedulers():
                scheduler.step()

    def on_test_epoch_end(self):
        out = {}
        for target in self.test_outputs.keys():
            y = torch.cat([y for y, _ in self.test_outputs[target]])
            y_hat = torch.cat([y_hat for _, y_hat in self.test_outputs[target]])
            out.update(self.calculate_metrics(y, y_hat, target))
        self.test_outputs = {k: [] for k in self.test_outputs.keys()}
        return out

    def calculate_log_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, target: str):
        if target == "label":
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
            self.log("val/label/bacc", bacc)
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            return loss, bacc
        if target in ["cf", "cf_std"]:
            mae = F.l1_loss(y_hat, y)
            mse = F.mse_loss(y_hat, y)
            r2 = r2_score(y.numpy(), y_hat.numpy())
            self.log("val/cf/mae", mae)
            self.log("val/cf/mse", mse)
            self.log("val/cf/r2", r2)
            return mse, r2
        else:
            raise ValueError(f"Unsupported target type: {target}")

    def calculate_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, target: str) -> dict:
        if target == "label":
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = roc_auc_score(y.cpu().numpy(), torch.sigmoid(y_hat).cpu().numpy())
            return {"label/bacc": bacc, "label/roc_auc": roc_auc}
        elif target in ["cf", "cf_std"]:
            mae = F.l1_loss(y_hat, y)
            mse = F.mse_loss(y_hat, y)
            r2 = r2_score(y.numpy(), y_hat.numpy())
            return {"cf/mae": mae, "cf/mse": mse, "cf/r2": r2}
        else:
            raise ValueError(f"Unsupported target type: {target}")


class cmmdRegularizedPredictor(MetaDataPrediction):
    def __init__(
        self,
        protected_attributes: list,  # e.g. ["cf"]
        cmmd_lambda: float = 1.0,
        inverted_bandwidth: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cmmd_lambda = cmmd_lambda
        self.inverted_bandwidth = inverted_bandwidth
        self.protected_attributes = protected_attributes

    def split_batch(self, batch: dict, attribute: str) -> list:
        """
        Splits the batch based on the given attribute.
        
        For our new data:
         - If attribute == "label": split into two dictionaries (label==0 and label==1).
         - If attribute == "cf": split using the median value.
        """
        if attribute == "label":
            mask = batch["label"] == 0
            return [
                {k: v[mask] for k, v in batch.items()},
                {k: v[~mask] for k, v in batch.items()}
            ]
        elif attribute in ["cf", "cf_std"]:
            # Compute median over the batch and split accordingly.
            median_val = 3.0
            mask = batch["cf"] < median_val
            return [
                {k: v[mask] for k, v in batch.items()},
                {k: v[~mask] for k, v in batch.items()}
            ]
        else:
            raise ValueError(f"Unknown attribute {attribute}")

    def get_combination_subgroups(self, batch: dict, attributes: list) -> list:
        """
        Recursively splits the batch for each attribute in the list.
        The result is the Cartesian product of the splits for each attribute.
        Note: We now check for the key "img" (instead of "mri").
        """
        subgroups = [batch]
        for attr in attributes:
            new_subgroups = []
            for subgroup in subgroups:
                splits = self.split_batch(subgroup, attr)
                new_subgroups.extend([
                    s for s in splits if s and s.get("img", None) is not None and len(s["img"]) > 0
                ])
            subgroups = new_subgroups
        return subgroups

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        x: torch.Tensor = batch["img"]  # new key
        y: torch.Tensor = batch[self.target]  # target is "label"
        y_hat: torch.Tensor = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        # Split by target "label"
        batch_splits_by_target = self.split_batch(batch, self.target)

        mmd_loss = 0.0
        added_elements = 0

        for y_group in batch_splits_by_target:
            if not y_group or len(y_group["img"]) == 0:
                continue
            # Further split by the protected attribute(s) (here: "cf")
            subgroup_list = self.get_combination_subgroups(y_group, self.protected_attributes)
            features_list = []
            for subgroup in subgroup_list:
                if subgroup["img"].shape[0] > 2:
                    features = self.forward(subgroup["img"])
                    features_list.append(features)
            if len(features_list) > 1:
                for i in range(len(features_list)):
                    for j in range(i + 1, len(features_list)):
                        mmd = mmd_compute(features_list[i], features_list[j], "gaussian", self.inverted_bandwidth)
                        mmd_loss += mmd
                        added_elements += 1

        if added_elements > 0:
            # mmd_loss /= added_elements
            loss = loss + mmd_loss * self.cmmd_lambda

        self.manual_backward(loss)
        optimizer.step()

        self.log("train/loss", loss.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["img"]
        y: torch.Tensor = batch[self.target]
        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        self.validation_outputs.append((y.detach().cpu(), y_hat.detach().cpu()))

        batch_splits_by_target = self.split_batch(batch, self.target)
        mmd_loss = 0.0
        added_elements = 0

        for y_group in batch_splits_by_target:
            if not y_group or len(y_group["img"]) == 0:
                continue
            subgroup_list = self.get_combination_subgroups(y_group, self.protected_attributes)
            features_list = []
            for subgroup in subgroup_list:
                if subgroup["img"].shape[0] > 2:
                    features = self.forward(subgroup["img"])
                    features_list.append(features)
            if len(features_list) > 1:
                for i in range(len(features_list)):
                    for j in range(i + 1, len(features_list)):
                        mmd = mmd_compute(features_list[i], features_list[j], "gaussian", self.inverted_bandwidth)
                        mmd_loss += mmd
                        added_elements += 1

        if added_elements > 0:
            # mmd_loss /= added_elements
            loss = loss + mmd_loss * self.cmmd_lambda

        self.log("val/mmd_loss", mmd_loss.detach().cpu() if added_elements > 0 else 0.0)
        self.log("val/full_loss", loss.detach().cpu())
        return loss
