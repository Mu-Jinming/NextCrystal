# path: src/predict_wyckoff.py
from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.predictors.wyckoff_predictor import (
    WyckoffPredictConfig,
    WyckoffSequencePredictor,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)

    pred_cfg = WyckoffPredictConfig(**OmegaConf.to_container(cfg.predict.wyckoff, resolve=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = WyckoffSequencePredictor(pred_cfg, device=device)
    output_path = predictor.save_predictions()

    logging.info("saved to %s", output_path)


if __name__ == "__main__":
    main()