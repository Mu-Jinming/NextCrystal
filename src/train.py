# path: src/train.py
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.data.sg_dataset import SGDataConfig, SGDataModule
from src.data.wyckoff_dataset import WyckoffDataConfig, WyckoffDataModule
from src.tasks.sg_classification_module import (
    SGClassificationLitModule,
    SGOptimizerConfig,
    SharedModelConfig,
)
from src.tasks.wyckoff_token_module import (
    WyckoffOptimizerConfig,
    WyckoffTokenLitModule,
)
from src.utils.inference_checkpoint import (
    extract_model_state_dict,
    save_inference_checkpoint,
)
from src.utils.wyckoff_template import (
    get_multiplicity_vocab_size,
    load_multiplicity_dict,
)


def build_logger(cfg):
    if not cfg.enabled:
        return None
    return WandbLogger(
        project=cfg.project,
        name=cfg.name,
        save_dir=cfg.save_dir,
        offline=cfg.offline,
        tags=cfg.tags,
        log_model=False,
    )


def export_best_inference_checkpoint(
    *,
    task: str,
    checkpoint_callback: ModelCheckpoint,
    output_dir: str,
    datamodule,
    model_cfg: SharedModelConfig,
) -> Path:
    best_path = checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError("Training finished without a best model checkpoint")

    lightning_checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = extract_model_state_dict(lightning_checkpoint)
    filename = "spacegroup.ckpt" if task == "sg" else "wyckoff.ckpt"
    checkpoint_task = "spacegroup" if task == "sg" else "wyckoff"
    output_path = Path(to_absolute_path(output_dir)) / filename

    best_score = checkpoint_callback.best_model_score
    metadata = {
        "source_checkpoint": str(Path(best_path).resolve()),
        "monitor": checkpoint_callback.monitor,
        "best_score": (
            float(best_score.detach().cpu()) if best_score is not None else None
        ),
    }
    if task == "sg":
        metadata["space_group_symbol_to_number"] = {
            str(symbol): int(number)
            for symbol, number in zip(
                datamodule.train_dataset.space_group_symbols,
                datamodule.train_dataset.space_group_numbers,
            )
        }
    return save_inference_checkpoint(
        output_path,
        task=checkpoint_task,
        state_dict=state_dict,
        formula_vocab=datamodule.formula_vocab,
        space_group_vocab=datamodule.space_group_vocab,
        wyckoff_vocab=(
            datamodule.wyckoff_vocab if task == "wyckoff" else None
        ),
        model_config=asdict(model_cfg),
        metadata=metadata,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    L.seed_everything(cfg.seed, workers=True)

    logger = build_logger(cfg.logger)
    callbacks = [LearningRateMonitor(logging_interval="epoch")]

    model_cfg = SharedModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    task = cfg.task

    if task == "sg":
        data_cfg = SGDataConfig(**OmegaConf.to_container(cfg.data.sg, resolve=True))
        optim_cfg = SGOptimizerConfig(
            lr=1e-3,
            weight_decay=1e-2,
            scheduler_factor=0.6,
            scheduler_patience=5,
            scheduler_min_lr=1e-6,
            monitor="val/acc",
        )

        datamodule = SGDataModule(data_cfg)
        datamodule.setup("fit")

        module = SGClassificationLitModule(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            vocab_size=len(datamodule.formula_vocab),
            space_group_size=len(datamodule.space_group_vocab),
            formula_vocab=datamodule.formula_vocab,
        )
        ckpt_monitor = "val/acc"
        ckpt_mode = "max"

    elif task == "wyckoff":
        data_cfg = WyckoffDataConfig(
            **OmegaConf.to_container(cfg.data.wyckoff, resolve=True)
        )
        optim_cfg = WyckoffOptimizerConfig(
            lr=1e-4,
            weight_decay=1e-2,
            betas=(0.9, 0.98),
            eps=1e-8,
            scheduler_factor=0.5,
            scheduler_patience=3,
            scheduler_min_lr=1e-6,
            scheduler_cooldown=1,
            scheduler_threshold=5e-4,
            monitor="val/loss",
            label_smoothing=0.1,
        )

        multiplicity_dict = load_multiplicity_dict(data_cfg.wyckoff_template_path)
        multiplicity_vocab_size = get_multiplicity_vocab_size(multiplicity_dict)

        datamodule = WyckoffDataModule(data_cfg, multiplicity_dict)
        datamodule.setup("fit")

        module = WyckoffTokenLitModule(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            vocab_size=len(datamodule.formula_vocab),
            space_group_size=len(datamodule.space_group_vocab),
            wyckoff_size=len(datamodule.wyckoff_vocab),
            multiplicity_vocab_size=multiplicity_vocab_size,
            pad_idx=datamodule.wyckoff_vocab["<pad>"],
            formula_vocab=datamodule.formula_vocab,
        )
        ckpt_monitor = "val/f1_macro"
        ckpt_mode = "max"

    else:
        raise ValueError(f"Unsupported task: {task}")

    checkpoint_callback = ModelCheckpoint(
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        save_top_k=1,
        save_last=True,
        filename="epoch={epoch:02d}",
    )
    callbacks.append(checkpoint_callback)

    if cfg.experiment.early_stopping_monitor is not None:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.experiment.early_stopping_monitor,
                mode=cfg.experiment.early_stopping_mode,
                patience=cfg.experiment.early_stopping_patience,
            )
        )

    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    trainer.fit(module, datamodule=datamodule)

    if trainer.is_global_zero:
        inference_checkpoint = export_best_inference_checkpoint(
            task=task,
            checkpoint_callback=checkpoint_callback,
            output_dir=cfg.inference_checkpoint_dir,
            datamodule=datamodule,
            model_cfg=model_cfg,
        )
        logging.info("Exported inference checkpoint to %s", inference_checkpoint)
    trainer.strategy.barrier("inference-checkpoint-export")

    if getattr(datamodule, "test_dataset", None) is not None:
        trainer.test(module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
