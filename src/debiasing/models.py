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
from pytorch_lightning.loggers import WandbLogger
from src.models.networks.resnet import ThreeDResNet
from src.models.networks.resnet_gn import ThreeDResNetGN
from src.models.networks.cnn import ThreeDCNN
from src.debiasing.cmmd_utils import mmd_compute

import hydra
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score


class CorrelationLossNegative(torch.nn.Module):
    def forward(self, inp: torch.Tensor, target: torch.Tensor):
        in_mean = inp.mean()
        tar_mean = target.mean()

        in_centered = inp - in_mean
        tar_centered = target - tar_mean

        r_numerator = torch.sum(in_centered * tar_centered)
        r_denominator = torch.sqrt((torch.sum(in_centered**2)) * torch.sum(tar_centered**2) + 1e-12) + 1e-12

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
        class_weights: Optional[np.ndarray | torch.Tensor] = None,
        predictor_type: str = "resnet",
    ):
        super().__init__()


        # adjust the number of outputs based on the target
        resnet_cfg["n_outputs"] = 1 if target in ["age","sex","age_std","all","age_bin"] else 3

        self.predictor_type = predictor_type
        if self.predictor_type == "resnet":
            self.predictor = ThreeDResNetGN(**resnet_cfg)
        elif self.predictor_type == "resnet_gn":
            self.predictor = ThreeDResNetGN(**resnet_cfg)
        elif self.predictor_type == "cnn":
            self.predictor = ThreeDCNN(**resnet_cfg)
        else:
            raise ValueError(f"Unsupported predictor type: {predictor_type}")

        self.lr_start = lr_start
        self.lr_end = lr_end
        self.target = target

        self.automatic_optimization = False

        self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

        self.save_hyperparameters(ignore=["predictor"])

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

        # if target is age_bin, replace the key age_std with age_bin
        if target == "age_bin":
            self.validation_outputs["age_bin"] = []
            self.test_outputs["age_bin"] = []

    def get_loss_function(self, target: str, weights: Optional[torch.Tensor] = None):
        if target in ["age", "age_std"]:
            return F.mse_loss
        elif target in ["sex", "age_bin"]:
            return BCEWithLogitsLoss(weight=weights)
        elif target == "site":
            return CrossEntropyLoss(weight=weights)
        else:
            raise ValueError(f"Unsupported target type: {target}")

    def forward(self, x) -> torch.Tensor:
        pass

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
            self.log("val/age_std/mae", mae)
            self.log("val/age_std/mse", mse)
            self.log("val/age_std/r2", r2)

            loss = mse
        # report balanced accuracy for sex
        elif target in ["sex", "age_bin"]:
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            self.log(f"val/{target}/bacc", bacc)

            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # report balanced accuracy for site
        elif target == "site":
            y_pred = y_hat.argmax(dim=1)
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            self.log("val/site/bacc", bacc)

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
        elif target in ["sex", "age_bin"]:
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
        pred_head: str = "linear",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.loss = self.get_loss_function(self.target, self.class_weights)
        self.pred_head = pred_head

        prediction_head = PredictionHeadLinear if pred_head == "linear" else PredictionHead

        self.classifier_task = prediction_head(
            self.predictor.feature_size, 
            1 if self.target in ["age", "age_std", "sex", "age_bin"] else 3
        )

        self.save_hyperparameters(ignore=["predictor"])

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.classifier_task(self.predictor(x)[1])

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
        optimizer = AdamW(list(self.predictor.parameters()) + list(self.classifier_task.parameters()), lr=self.lr_start)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr_end)
        return [optimizer], [scheduler]

    def on_validation_epoch_end(self):
        # aggregate predictions
        y = torch.cat([y for y, _ in self.validation_outputs])
        y_hat = torch.cat([y_hat for _, y_hat in self.validation_outputs])

        self.calculate_log_metrics(y, y_hat, self.target)

        self.validation_outputs = []

        # do scheduler step if it is not a sanity check
        if not self.trainer.sanity_checking:
            self.lr_schedulers().step()

    def on_test_epoch_end(self):
        # aggregate predictions
        y = torch.cat([y for y, _ in self.test_outputs])
        y_hat = torch.cat([y_hat for _, y_hat in self.test_outputs])
        
        metrics = self.calculate_metrics(y, y_hat, self.target)
        self.test_outputs = []
        return metrics


class AdversarialPredictor(MetaDataPredictionAbstract):
    def __init__(
        self,
        protected_attributes: List[str],
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

        prediction_head = PredictionHeadLinear if protected_pred_head == "linear" else PredictionHead

        self.protected_losses = {k: self.get_loss_function(k) for k in protected_attributes}

        self.feature_losses = {k: self.get_loss_function_feature(k) for k in protected_attributes}
        
        self.pred_heads_protected = torch.nn.ModuleDict(
            {
                k: prediction_head(
                    self.predictor.feature_size, 
                    1 if k in ["age", "sex", "age_std", "all", "age_bin"] else 3
                ) 
                for k in protected_attributes
            }
        )

        self.classifier_task = prediction_head(
            self.predictor.feature_size, 
            1 if self.target in ["age", "sex", "age_std", "age_bin"] else 3
        )

        self.save_hyperparameters(ignore=["predictor", "pred_heads"])


    def get_loss_function_feature(self, target: str, weights: Optional[torch.Tensor] = None):
        if target in ["age", "age_std"]:
            return CorrelationLossNegative()
        elif target in ["sex", "age_bin"]:
            return BCEWithLogitsLoss(weight=weights)
        elif target == "site":
            return CrossEntropyLoss(weight=weights)
        else:
            raise ValueError(f"Unsupported target type: {target}")

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

    def split_batch(self, batch) -> List[Dict[str, torch.Tensor]]:
        """a batch is a dict with keys ["mri", "age", "sex", "site", "age_std", etc.]
        and each key has a value of shape (batch_size, ...).
        this function returns new dicts, depending on the target. if target is age_std, then
        split the batch into two dicts, s.t. batch one has age_std smaller 0.0 and batch two has age_std >= 0.0.
        if target is sex, return batches such that sex==0 and sex==1. For site, site==0 and site==1 and site==2.
        """
        if self.target == "age_std":
            mask = batch["age_std"] < 0.0
            return [{k: v[mask] for k, v in batch.items()}, {k: v[~mask] for k, v in batch.items()}]
        elif self.target in ["sex", "age_bin"]:
            mask = batch[self.target] == 0
            return [{k: v[mask] for k, v in batch.items()}, {k: v[~mask] for k, v in batch.items()}]
        elif self.target == "site":
            mask1 = batch["site"] == 0
            mask2 = batch["site"] == 1
            mask3 = batch["site"] == 2
            return [{k: v[mask1] for k, v in batch.items()}, {k: v[mask2] for k, v in batch.items()}, {k: v[mask3] for k, v in batch.items()}]

    def training_step(self, batch, batch_idx):

        optimizer_full, optimizer_features, *optimizer_protected = self.optimizers()
        optimizer_full.zero_grad()
        optimizer_features.zero_grad()
        for o in optimizer_protected:
            o.zero_grad()

        x: torch.Tensor = batch["mri"]
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
            if not b or len(b["mri"]) == 0:
                continue
            x = b["mri"]
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
        x: torch.Tensor = batch["mri"]
        y: torch.Tensor = batch[self.target]

        features = self.predictor(x)[1]
        y_hat = self.classifier_task(features).squeeze()
        loss = self.loss(y_hat.squeeze(), y)

        self.validation_outputs[self.target].append((y.detach().cpu(), y_hat.detach().cpu()))

        for protected_attribute in self.protected_attributes:
            y_protected = batch[protected_attribute]
            y_hat_protected = self.pred_heads_protected[protected_attribute](features).squeeze()
            loss_protected = self.protected_losses[protected_attribute](y_hat_protected, y_protected)
            self.validation_outputs[protected_attribute].append((y_protected.detach().cpu(), y_hat_protected.detach().cpu()))

        return loss

    def test_step(self, batch, batch_idx):
        x: torch.Tensor = batch["mri"]
        y: torch.Tensor = batch[self.target]

        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat.squeeze(), y)

        self.test_outputs[self.target].append((y.detach().cpu(), y_hat.detach().cpu()))

        # for protected_attribute in self.protected_attributes:
        #     y_protected = batch[protected_attribute]
        #     y_hat_protected = self.pred_heads_protected[protected_attribute](self.predictor(x)[1]).squeeze()
        #     loss_protected = self.protected_losses[protected_attribute](y_hat_protected, y_protected)
        #     self.test_outputs[protected_attribute].append((y_protected.detach().cpu(), y_hat_protected.detach().cpu()))

        return loss

    def on_validation_epoch_end(self):
        loss = 0.0
        for target in [k for k, v in self.validation_outputs.items() if len(v) > 0]:
            y = torch.cat([y for y, _ in self.validation_outputs[target]])
            y_hat = torch.cat([y_hat for _, y_hat in self.validation_outputs[target]])

            l, m = self.calculate_log_metrics(y, y_hat, target)
            if target == self.target:
                loss = l
        
        # especially for checkpointing
        self.log("val/loss", loss)
        self.validation_outputs = {k: [] for k in self.validation_outputs.keys()}

        # do scheduler step if it is not a sanity check
        if not self.trainer.sanity_checking:
            for scheduler in self.lr_schedulers():
                scheduler.step()

    def on_test_epoch_end(self):
        out = {}
        for target in [k for k, v in self.test_outputs.items() if len(v) > 0]:
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
        elif target in ["sex", "age_bin"]:
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            self.log(f"val/{target}/bacc", bacc)

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
        elif target in ["sex", "age_bin"]:
            y_pred = (y_hat > 0).float()
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            roc_auc = roc_auc_score(y.numpy(), torch.sigmoid(y_hat).numpy())
            return {f"{target}/bacc": bacc, f"{target}/roc_auc": roc_auc}
        # report balanced accuracy for site
        elif target == "site":
            y_pred = y_hat.argmax(dim=1)
            bacc = balanced_accuracy_score(y.numpy(), y_pred.numpy())
            roc_auc = roc_auc_score(y.numpy(), F.softmax(y_hat, dim=1).numpy(), multi_class="ovo")
            return {"site/bacc": bacc, "site/roc_auc": roc_auc}
        else:
            raise ValueError(f"Unsupported target type: {target}")


class cmmdRegularizedPredictor(MetaDataPrediction):
    def __init__(
        self,
        protected_attributes: List[str],
        cmmd_lambda: float = 1.0,
        inverted_bandwidth: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cmmd_lambda = cmmd_lambda
        self.inverted_bandwidth = inverted_bandwidth
        self.protected_attributes = protected_attributes

    def split_batch(self, batch: dict, attribute: str) -> List[Dict[str, torch.Tensor]]:
        """
        Splits the batch based on the given attribute.
        
        For continuous attribute 'age_std', we split by negative vs non-negative.
        For categorical attributes such as 'sex' and 'site', we split by each category.
        """
        if attribute == "age_std":
            mask = batch["age_std"] < 0.0
            return [{k: v[mask] for k, v in batch.items()}, {k: v[~mask] for k, v in batch.items()}]
        elif attribute in ["sex", "age_bin"]:
            mask = batch[attribute] == 0
            return [{k: v[mask] for k, v in batch.items()}, {k: v[~mask] for k, v in batch.items()}]
        elif attribute == "site":
            # Assuming sites are coded as 0, 1, 2.
            splits = []
            for site_val in [0, 1, 2]:
                mask = batch["site"] == site_val
                splits.append({k: v[mask] for k, v in batch.items()})
            return splits
        else:
            raise ValueError(f"Unknown attribute {attribute}")

    def get_combination_subgroups(self, batch: dict, attributes: List[str]) -> List[dict]:
        """
        Recursively splits the batch for each attribute in the list.
        The result is the Cartesian product of the splits for each attribute.
        """
        subgroups = [batch]
        for attr in attributes:
            new_subgroups = []
            for subgroup in subgroups:
                splits = self.split_batch(subgroup, attr)
                # Only keep splits that have samples
                new_subgroups.extend([s for s in splits if s and s.get("mri", None) is not None and len(s["mri"]) > 0])
            subgroups = new_subgroups
        return subgroups

    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        optimizer.zero_grad()

        x: torch.Tensor = batch["mri"]
        y: torch.Tensor = batch[self.target]

        y_hat: torch.Tensor = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        # First, split by target Y. We want to compute MMD only within each y.
        batch_splits_by_target: List[dict] = self.split_batch(batch, self.target)

        mmd_loss = 0.0
        added_elements = 0

        for y_group in batch_splits_by_target:
            # Check if there is at least one sample in this target group
            if not y_group or len(y_group["mri"]) == 0:
                continue

            # Get all subgroups for all protected attribute combinations within this Y=y group.
            subgroup_list = self.get_combination_subgroups(y_group, self.protected_attributes)
            # Optionally, print sizes for debugging:
            # for idx, subgroup in enumerate(subgroup_list):
            #     print(f"Subgroup {idx} for y group has {len(subgroup['mri'])} samples.")

            # Pre-compute features for each subgroup that has enough samples (at least 3)
            features_list = []
            for subgroup in subgroup_list:
                if subgroup["mri"].shape[0] > 2:
                    features = self.forward(subgroup["mri"])
                    features_list.append(features)
            # Compute MMD between all pairs of subgroups in this Y group.
            if len(features_list) > 1:
                for i in range(len(features_list)):
                    for j in range(i + 1, len(features_list)):
                        mmd = mmd_compute(features_list[i], features_list[j], "gaussian", self.inverted_bandwidth)
                        mmd_loss += mmd
                        added_elements += 1

        if added_elements > 0:
            mmd_loss /= added_elements
            loss += mmd_loss * self.cmmd_lambda

        self.manual_backward(loss)
        optimizer.step()

        self.log("train/loss", loss.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["mri"]
        y: torch.Tensor = batch[self.target]

        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat.squeeze(), y)

        self.validation_outputs.append((y.detach().cpu(), y_hat.detach().cpu()))

        batch_splits_by_target: List[dict] = self.split_batch(batch, self.target)

        mmd_loss = 0.0
        added_elements = 0

        for y_group in batch_splits_by_target:
            # Check if there is at least one sample in this target group
            if not y_group or len(y_group["mri"]) == 0:
                continue

            # Get all subgroups for all protected attribute combinations within this Y=y group.
            subgroup_list = self.get_combination_subgroups(y_group, self.protected_attributes)
            # Optionally, print sizes for debugging:
            # for idx, subgroup in enumerate(subgroup_list):
            #     print(f"Subgroup {idx} for y group has {len(subgroup['mri'])} samples.")

            # Pre-compute features for each subgroup that has enough samples (at least 3)
            features_list = []
            for subgroup in subgroup_list:
                if subgroup["mri"].shape[0] > 2:
                    features = self.forward(subgroup["mri"])
                    features_list.append(features)
            # Compute MMD between all pairs of subgroups in this Y group.
            if len(features_list) > 1:
                for i in range(len(features_list)):
                    for j in range(i + 1, len(features_list)):
                        mmd = mmd_compute(features_list[i], features_list[j], "gaussian", self.inverted_bandwidth)
                        mmd_loss += mmd
                        added_elements += 1

        if added_elements > 0:
            mmd_loss /= added_elements
            loss += mmd_loss * self.cmmd_lambda

        self.log("val/mmd_loss", mmd_loss.detach().cpu() if added_elements > 0 else 0.0)
        self.log("val/full_loss", loss.detach().cpu())
        return loss
