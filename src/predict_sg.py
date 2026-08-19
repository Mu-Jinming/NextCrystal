# path: src/predict_sg.py
from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.predictors.sg_predictor import SGPredictConfig, SpaceGroupPredictor


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)

    pred_cfg = SGPredictConfig(**OmegaConf.to_container(cfg.predict.sg, resolve=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = SpaceGroupPredictor(pred_cfg, device=device)
    output_path = predictor.save_predictions()

    logging.info("saved to %s", output_path)


if __name__ == "__main__":
    main()