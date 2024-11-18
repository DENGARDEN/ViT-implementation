import time

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import DeviceStatsMonitor, EarlyStopping, Timer
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import AUROC
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.models import ResNet  # For comparison


class ClassificationLightningModel(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        lr,
        weight_decay=0.0,
        warmup_epochs=20,
        num_epochs=20,
    ):
        super().__init__()
        if isinstance(model, ResNet):
            model.fc = nn.Linear(
                model.fc.in_features, num_classes
            )  # Change the last layer to output num_classes

        self.model = model
        self.num_classes = num_classes

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Basic metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

        # Additional classification metrics
        self.test_f1 = MulticlassF1Score(num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)

        # Time tracking
        self.inference_times = []

        # Add storage for predictions
        self.all_test_preds = []
        self.all_test_targets = []

        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.softmax(logits, dim=1)
        self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.softmax(logits, dim=1)
        self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Measure inference time
        start_time = time.time()
        logits = self(x)
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        loss = self.criterion(logits, y)
        preds = torch.softmax(logits, dim=1)

        # Store predictions and targets
        self.all_test_preds.append(preds)
        self.all_test_targets.append(y)

        # Calculate metrics
        self.test_acc.update(preds, y)
        self.test_f1.update(logits, y)  # F1 uses logits
        self.test_auroc.update(preds, y)  # AUROC uses probabilities

        return {
            "loss": loss,
            "batch_preds": preds.detach(),
            "batch_targets": y.detach(),
        }

    def on_test_epoch_end(self):

        # Compute final metrics
        test_acc = self.test_acc.compute()
        test_f1 = self.test_f1.compute()
        test_auroc = self.test_auroc.compute()

        # Log metrics
        self.log("test_acc", test_acc)
        self.log("test_f1", test_f1)
        self.log("test_auroc", test_auroc)

        # Combine  predictions and targets
        all_preds = torch.cat(self.all_test_preds)
        all_targets = torch.cat(self.all_test_targets)

        # Calculate and log average inference time

        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        self.log("avg_inference_time", avg_inference_time)

        # Calculate model size
        model_size = sum(p.numel() for p in self.model.parameters())
        self.log("model_size", model_size)

        # Save predictions to file
        predictions_dict = {
            "predictions": all_preds.cpu().numpy(),
            "targets": all_targets.cpu().numpy(),
            "metrics": {
                "accuracy": test_acc.item(),
                "f1_score": test_f1.item(),
                "auroc": test_auroc.item(),
                "avg_inference_time": avg_inference_time,
            },
        }
        np.save("./temp/test_predictions.npy", predictions_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(  # Using AdamW instead of Adam
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Calculate total steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * (self.warmup_epochs / self.trainer.max_epochs))

        # Create warmup scheduler using LinearLR
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,  # Start from 0
            end_factor=1.0,  # Reach full lr
            total_iters=warmup_steps,
        )

        # Main scheduler using CosineAnnealingLR
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=0.0
        )

        # Combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# Setup training with metrics monitoring
def train_and_evaluate_model(model, num_classes, train_loader, val_loader, test_loader, **kwargs):
    # Initialize model with metrics

    epochs = kwargs.get("epochs", 20)
    lr = kwargs.get("lr", 1e-3)

    weight_decay = kwargs.get("weight_decay", 0.0)
    model_name = kwargs.get("model_name", "proj")
    patience = kwargs.get("patience", 5)
    warmup_epochs = kwargs.get("warmup_epochs", 20)

    lit_model = ClassificationLightningModel(
        model,
        num_classes=num_classes,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        num_epochs=epochs,
    )

    # Callbacks for monitoring
    timer = Timer()
    device_stats = DeviceStatsMonitor()
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience)

    # Initialize trainer with monitoring
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[timer, device_stats, early_stopping],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=TensorBoardLogger(save_dir="logs/", name=model_name),
    )

    # Train and test
    trainer.fit(lit_model, train_loader, val_loader)
    test_results = trainer.test(lit_model, test_loader)

    try:
        predictions_dict = np.load(
            "./temp/test_predictions.npy", allow_pickle=True  # NEW: Dynamic filename
        ).item()

        metrics = {
            "test_accuracy": float(lit_model.test_acc.compute()),
            "training_time": timer.time_elapsed("train"),
            "model_size": sum(p.numel() for p in model.parameters()),
            "avg_inference_time": predictions_dict["metrics"]["avg_inference_time"],
            "test_f1": float(lit_model.test_f1.compute()),
            "test_auroc": float(lit_model.test_auroc.compute()),
            "predictions": predictions_dict["predictions"],
            "targets": predictions_dict["targets"],
        }
    except Exception as e:
        print(f"Warning: Could not load predictions file: {e}")
        metrics = {
            "test_accuracy": float(lit_model.test_acc.compute()),
            "training_time": timer.time_elapsed("train"),
            "model_size": sum(p.numel() for p in model.parameters()),
            "test_f1": float(lit_model.test_f1.compute()),
            "test_auroc": float(lit_model.test_auroc.compute()),
        }

    return metrics
