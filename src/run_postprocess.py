# path: src/run_postprocess.py
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.postprocess.wyckoff_assignment import (
    PostProcessConfig,
    WyckoffAssignmentPostProcessor,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)

    post_cfg = PostProcessConfig(**OmegaConf.to_container(cfg.postprocess, resolve=True))
    processor = WyckoffAssignmentPostProcessor(post_cfg)
    processor.run()

    logging.info("saved to %s", post_cfg.output_csv)


if __name__ == "__main__":
    main()