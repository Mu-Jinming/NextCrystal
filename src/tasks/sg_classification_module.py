# path: src/tasks/sg_classification_module.py
from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import torch
import torch.nn as nn
import torchmetrics

from src.models.wyckoff_predictor import WyckoffPredictor


@dataclass
class SharedModelConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_atoms: int = 512
    max_multiplicities: int = 30

    use_moe: bool = True
    moe_layers: str | list[int] = "all"
    num_experts: int = 8
    capacity_factor: float = 1.5
    router_noisy_std: float = 0.5
    moe_loss_coef: float = 0.0
    moe_top_k: int = 2


@dataclass
class SGOptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    scheduler_factor: float = 0.6
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    monitor: str = "val/acc"


class SGClassificationLitModule(L.LightningModule):
    def __init__(self, model_cfg: SharedModelConfig, optim_cfg: SGOptimizerConfig, vocab_size: int, space_group_size: int, formula_vocab):
        super().__init__()
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg

        self.model = WyckoffPredictor(
            vocab_size=vocab_size,
            space_group_size=space_group_size,
            wyckoff_size=2,
            multiplicity_vocab_size=1,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            num_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            max_len=model_cfg.max_atoms,
            dropout=model_cfg.dropout,
            max_multiplicities=model_cfg.max_multiplicities,
            use_moe=model_cfg.use_moe,
            moe_layers=model_cfg.moe_layers,
            num_experts=model_cfg.num_experts,
            capacity_factor=model_cfg.capacity_factor,
            router_noisy_std=model_cfg.router_noisy_std,
            moe_loss_coef=model_cfg.moe_loss_coef,
            moe_top_k=model_cfg.moe_top_k,
            formula_vocab=formula_vocab,
            enable_sg_head=True,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=space_group_size)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=space_group_size)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=space_group_size)

    def forward(self, tokens, mask):
        return self.model.forward_sg(tokens, mask)

    def _shared_step(self, batch, stage: str):
        logits, aux_loss = self(batch["tokens"], batch["mask"])
        loss = self.criterion(logits, batch["sg_label"]) + self.model.moe_loss_coef * aux_loss
        preds = logits.argmax(-1)

        metric = {"train": self.train_acc, "val": self.val_acc, "test": self.test_acc}[stage]
        acc = metric(preds, batch["sg_label"])

        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=(stage != "test"), batch_size=batch["tokens"].size(0))
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["tokens"].size(0))
        self.log(f"{stage}/moe_aux_loss", aux_loss, on_step=False, on_epoch=True, batch_size=batch["tokens"].size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.optim_cfg.scheduler_factor,
            patience=self.optim_cfg.scheduler_patience,
            min_lr=self.optim_cfg.scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.optim_cfg.monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }