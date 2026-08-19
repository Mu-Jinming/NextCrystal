# path: src/tasks/wyckoff_token_module.py
from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import precision_score, recall_score, f1_score

from src.models.wyckoff_predictor import WyckoffPredictor


@dataclass
class WyckoffOptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6
    scheduler_cooldown: int = 1
    scheduler_threshold: float = 5e-4
    monitor: str = "val/loss"
    label_smoothing: float = 0.1


class WyckoffTokenLitModule(L.LightningModule):
    def __init__(
        self,
        model_cfg,
        optim_cfg: WyckoffOptimizerConfig,
        vocab_size: int,
        space_group_size: int,
        wyckoff_size: int,
        multiplicity_vocab_size: int,
        pad_idx: int,
        formula_vocab,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.pad_idx = pad_idx

        self.model = WyckoffPredictor(
            vocab_size=vocab_size,
            space_group_size=space_group_size,
            wyckoff_size=wyckoff_size,
            multiplicity_vocab_size=multiplicity_vocab_size,
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
            enable_sg_head=False,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=optim_cfg.label_smoothing)
        self.val_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=wyckoff_size, ignore_index=pad_idx)
        self.val_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=wyckoff_size, top_k=5, ignore_index=pad_idx)

    def forward(self, batch):
        return self.model(
            batch["tokens"],
            batch["mask"],
            batch["space_group"],
            batch["multiplicities"],
            batch["multiplicity_mask"],
        )

    def _token_loss(self, logits, labels):
        return self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

    def training_step(self, batch, batch_idx):
        logits, aux_loss = self(batch)
        ce = self._token_loss(logits, batch["labels"])
        loss = ce + self.model.moe_loss_coef * aux_loss

        preds = logits.argmax(-1)
        valid = batch["mask"]
        acc = ((preds == batch["labels"]) & valid).sum().float() / valid.sum().clamp(min=1)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["tokens"].size(0))
        self.log("train/ce_loss", ce, on_step=False, on_epoch=True, batch_size=batch["tokens"].size(0))
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["tokens"].size(0))
        self.log("train/moe_aux_loss", aux_loss, on_step=False, on_epoch=True, batch_size=batch["tokens"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        logits, aux_loss = self(batch)
        ce = self._token_loss(logits, batch["labels"])
        loss = ce + self.model.moe_loss_coef * aux_loss

        preds = logits.argmax(-1)
        labels = batch["labels"]

        self.val_top1.update(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.val_top5.update(logits.view(-1, logits.size(-1)), labels.view(-1))

        valid = batch["mask"]
        pred_np = preds[valid].detach().cpu().numpy()
        label_np = labels[valid].detach().cpu().numpy()

        precision = precision_score(label_np, pred_np, average="macro", zero_division=0) if len(label_np) else 0.0
        recall = recall_score(label_np, pred_np, average="macro", zero_division=0) if len(label_np) else 0.0
        f1 = f1_score(label_np, pred_np, average="macro", zero_division=0) if len(label_np) else 0.0

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["tokens"].size(0))
        self.log("val/ce_loss", ce, on_step=False, on_epoch=True, batch_size=batch["tokens"].size(0))
        self.log("val/moe_aux_loss", aux_loss, on_step=False, on_epoch=True, batch_size=batch["tokens"].size(0))
        self.log("val/precision_macro", precision, on_step=False, on_epoch=True, batch_size=batch["tokens"].size(0))
        self.log("val/recall_macro", recall, on_step=False, on_epoch=True, batch_size=batch["tokens"].size(0))
        self.log("val/f1_macro", f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["tokens"].size(0))

    def on_validation_epoch_end(self):
        self.log("val/acc_top1", self.val_top1.compute(), prog_bar=True)
        self.log("val/acc_top5", self.val_top5.compute(), prog_bar=True)
        self.val_top1.reset()
        self.val_top5.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_cfg.lr,
            weight_decay=self.optim_cfg.weight_decay,
            betas=self.optim_cfg.betas,
            eps=self.optim_cfg.eps,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.optim_cfg.scheduler_factor,
            patience=self.optim_cfg.scheduler_patience,
            min_lr=self.optim_cfg.scheduler_min_lr,
            cooldown=self.optim_cfg.scheduler_cooldown,
            threshold=self.optim_cfg.scheduler_threshold,
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
