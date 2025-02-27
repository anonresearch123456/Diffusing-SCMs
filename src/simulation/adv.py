from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl

from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from src.simulation.resnet import ConvNet

# (1) Define a custom encoder that exactly replicates the Keras feature encoder:
#
# Keras encoder:
#   Input: 32×32×1
#   Conv2D(2, kernel_size=3, activation=tanh, padding='valid')   --> 30×30×2
#   MaxPool2D(pool_size=2)                                         --> 15×15×2
#   Conv2D(4, kernel_size=3, activation=tanh, padding='valid')       --> 13×13×4
#   MaxPool2D(pool_size=2)                                         --> 6×6×4
#   Conv2D(8, kernel_size=3, activation=tanh, padding='valid')       --> 4×4×8
#   MaxPool2D(pool_size=2)                                         --> 2×2×8
#   Flatten()                                                    --> 32
#
class CustomEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=0)  # valid padding
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.act = torch.tanh  # using the built-in tanh

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return x
    

class PredictionHeadLinear(torch.nn.Module):
    def __init__(self, in_features: int, n_outputs: int):
        super().__init__()
        # use layers with ReLU activation
        self.fc = torch.nn.Linear(in_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# (2) Define the classifier head (mimicking Keras: Dense(16, tanh) then Dense(1))
class ClassifierHead(nn.Module):
    def __init__(self, in_features=32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.fc2 = nn.Linear(16, 1)
        self.act = torch.tanh

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

# (3) Define the regressor (bias predictor) as in the Keras code: Dense(16, tanh) then Dense(1)
class Regressor(nn.Module):
    def __init__(self, in_features=32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.fc2 = nn.Linear(16, 1)
        self.act = torch.tanh

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

# (4) Use the already existing correlation loss (which computes r^2) from your ecosystem.
# (If you prefer to use your own version, see below.)
class CorrelationLoss(nn.Module):
    def forward(self, inp: torch.Tensor, target: torch.Tensor):
        in_mean = inp.mean()
        tar_mean = target.mean()
        in_centered = inp - in_mean
        tar_centered = target - tar_mean
        r_numerator = torch.sum(in_centered * tar_centered)
        r_denominator = torch.sqrt(torch.sum(in_centered ** 2) * torch.sum(tar_centered ** 2)) + 1e-5
        r = r_numerator / r_denominator
        r = torch.clamp(r, min=-1.0, max=1.0)
        return r ** 2

# (5) Now, create the Lightning module that implements the adversarial training.
#
# It sets up three optimizers:
#   - optimizer_regressor (lr=0.0002): updates the regressor only (with the encoder “frozen”)
#   - optimizer_distiller (lr=0.0002): updates the encoder so as to minimize the correlation
#   - optimizer_classifier (lr=0.0001): updates the encoder and classifier for the classification task.
#
# In each training_step we:
#   (a) sample a “control” batch (samples with label == 0) for the regressor and distiller steps;
#   (b) update the regressor on features computed with the encoder (without backpropagating to the encoder);
#   (c) update the encoder (“distiller”) to minimize the correlation between cf and the regressor’s output;
#   (d) update the encoder and classifier with the classification loss.
#
class AdversarialMetaDataPrediction(pl.LightningModule):
    def __init__(self, lr_classifier=1e-4, lr_regressor=2e-4, lr_distiller=2e-4,
                 **kwargs):
        """
        This module implements the adversarial training pipeline equivalent to the research Keras code.
        It uses:
          - a custom encoder (feature extractor) matching the Keras conv/maxpool architecture,
          - a regressor (bias predictor) and a distiller (adversarially removing bias from features),
          - and a classifier (for the actual prediction task).
        """
        super().__init__()
        self.automatic_optimization = False  # we handle three optimizers manually

        # Build the networks
        self.encoder = ConvNet(in_channels=1, n_outputs=1, depth=4, n_basefilters=8)
        self.regressor = PredictionHeadLinear(self.encoder.feature_size, 1)
        self.classifier = PredictionHeadLinear(self.encoder.feature_size, 1)

        # Learning rates (match Keras: classifier 1e-4; regressor & distiller 2e-4)
        self.lr_classifier = lr_classifier
        self.lr_regressor = lr_regressor
        self.lr_distiller = lr_distiller

        # Loss functions
        self.criterion_bce = nn.BCEWithLogitsLoss()  # for classification
        self.criterion_mse = nn.MSELoss()            # for regressor
        self.criterion_corr = CorrelationLoss()        # for distiller

        self.validation_outputs: Dict[str, List[Tuple[torch.Tensor]]] = {
            "label": [],
            "cf": []
        }
        self.test_outputs: Dict[str, List[Tuple[torch.Tensor]]] = {
            "label": [],
            "cf": []
        }

    def forward(self, x):
        # For inference: pass through encoder then classifier.
        _, features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]    # shape: [B, 1, 32, 32]
        labels = batch["label"]  # binary classification (0: control, 1: disease)
        cf = batch["cf"]       # confounder (bias)

        # --- 1. Train the regressor (bias predictor) using a control-group batch ---
        # select only control samples (label==0)
        ctrl_mask = (labels == 0)
        # if no control samples in batch, use the entire batch (as a fallback)
        if ctrl_mask.sum() < 1:
            ctrl_mask = torch.ones_like(labels, dtype=torch.bool)
        imgs_ctrl = imgs[ctrl_mask]
        cf_ctrl = cf[ctrl_mask]

        # To update the regressor only, compute features without backprop to encoder.
        self.encoder.eval()
        with torch.no_grad():
            _, features_ctrl = self.encoder(imgs_ctrl)
        self.encoder.train()  # restore training mode

        cf_pred = self.regressor(features_ctrl).squeeze()  # predicted cf
        loss_regressor = self.criterion_mse(cf_pred, cf_ctrl)

        optimizer_regressor = self.optimizers()[0]
        optimizer_regressor.zero_grad()
        self.manual_backward(loss_regressor)
        optimizer_regressor.step()

        # --- 2. Train the distiller (adversarial bias removal) ---
        # Freeze regressor parameters so that only the encoder is updated.
        for param in self.regressor.parameters():
            param.requires_grad = False

        _, features_ctrl = self.encoder(imgs_ctrl)  # encoder is trainable here
        cf_pred_distill = self.regressor(features_ctrl).squeeze()
        loss_distiller = self.criterion_corr(cf_pred_distill, cf_ctrl)

        optimizer_distiller = self.optimizers()[1]
        optimizer_distiller.zero_grad()
        self.manual_backward(loss_distiller)
        optimizer_distiller.step()

        # Unfreeze the regressor parameters.
        for param in self.regressor.parameters():
            param.requires_grad = True

        # --- 3. Train the classifier (for the actual classification task) ---
        _, features = self.encoder(imgs)
        logits = self.classifier(features).squeeze()
        loss_classifier = self.criterion_bce(logits, labels)

        optimizer_classifier = self.optimizers()[2]
        optimizer_classifier.zero_grad()
        self.manual_backward(loss_classifier)
        optimizer_classifier.step()

        # Log the three losses
        self.log("train/loss_regressor", loss_regressor)
        self.log("train/loss_distiller", loss_distiller)
        self.log("train/loss_classifier", loss_classifier)

        # You can also return a dict for debugging
        return {
            "loss_classifier": loss_classifier,
            "loss_regressor": loss_regressor,
            "loss_distiller": loss_distiller
        }

    def validation_step(self, batch, batch_idx):
        imgs = batch["img"]
        labels = batch["label"]
        cf = batch["cf"]
        _, features = self.encoder(imgs)
        logits = self.classifier(features).squeeze()
        loss = self.criterion_bce(logits, labels)
        self.log("val/loss", loss, prog_bar=True)

        self.validation_outputs["label"].append((labels.detach().cpu(), logits.detach().cpu()))
        self.validation_outputs["cf"].append((cf.detach().cpu(), self.regressor(features).detach().cpu()))
        return loss

    def configure_optimizers(self):
        # Return three separate optimizers (the ordering is important in training_step):
        #   0: regressor optimizer (lr=2e-4)
        #   1: distiller optimizer (encoder update, lr=2e-4)
        #   2: classifier optimizer (encoder + classifier update, lr=1e-4)
        optimizer_regressor = Adam(self.regressor.parameters(), lr=self.lr_regressor)
        optimizer_distiller = Adam(self.encoder.parameters(), lr=self.lr_distiller)
        optimizer_classifier = Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=self.lr_classifier
        )
        return [optimizer_regressor, optimizer_distiller, optimizer_classifier]

    def on_validation_epoch_end(self):
        # Compute validation metrics
        labels = torch.cat([x[0] for x in self.validation_outputs["label"]])
        logits = torch.cat([x[1] for x in self.validation_outputs["label"]])
        cf = torch.cat([x[0] for x in self.validation_outputs["cf"]])
        cf_pred = torch.cat([x[1] for x in self.validation_outputs["cf"]])

        val_acc = balanced_accuracy_score(labels.numpy(), (logits > 0).numpy())
        val_r2 = r2_score(cf.numpy(), cf_pred.numpy())
        self.log("val/label/bacc", val_acc)
        self.log("val/cf/r2", val_r2)

        # Reset the outputs for the next epoch
        self.validation_outputs = {
            "label": [],
            "cf": []
        }