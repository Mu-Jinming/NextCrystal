# path: src/data/sg_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.chemistry import build_formula_vocab, formula_to_tokens
from src.utils.wyckoff_template import (
    build_space_group_vocab,
    space_group_number_to_index,
)


@dataclass
class SGDataConfig:
    train_csv: str
    val_csv: str
    test_csv: Optional[str] = None
    wyckoff_template_path: str = "data/wyckoff_template.json"
    formula_column: str = "formula"
    natoms_column: str = "NAtoms"
    sg_symbol_column: str = "Spacegroup Symbol"
    sg_number_column: str = "Spacegroup Number"
    max_atoms: int = 512
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False


class SGDataset(Dataset):
    def __init__(
        self,
        formulas,
        n_atoms,
        space_group_symbols,
        space_group_numbers,
        formula_vocab=None,
        space_group_vocab=None,
        wyckoff_template_path: str = "data/wyckoff_template.json",
        max_atoms: int = 512,
    ):
        self.formulas = list(formulas)
        self.n_atoms = [int(x) for x in n_atoms]
        self.space_group_symbols = list(space_group_symbols)
        self.space_group_numbers = [int(x) for x in space_group_numbers]
        self.max_atoms = max_atoms
        self.wyckoff_template_path = wyckoff_template_path

        self.formula_vocab = formula_vocab or build_formula_vocab(self.formulas)
        self.space_group_vocab = (
            space_group_vocab
            if space_group_vocab is not None
            else self.build_space_group_vocab()
        )

    def build_space_group_vocab(self):
        return build_space_group_vocab(self.wyckoff_template_path)

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        tokens, mask = formula_to_tokens(
            formula=self.formulas[idx],
            n_atoms=self.n_atoms[idx],
            formula_vocab=self.formula_vocab,
            max_atoms=self.max_atoms,
        )
        sg_idx = space_group_number_to_index(self.space_group_numbers[idx])
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "sg_label": torch.tensor(sg_idx, dtype=torch.long),
        }


def sg_collate_fn(batch):
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "sg_label": torch.stack([item["sg_label"] for item in batch]),
    }


class SGDataModule(L.LightningDataModule):
    def __init__(self, cfg: SGDataConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.formula_vocab = None
        self.space_group_vocab = None

    def setup(self, stage: Optional[str] = None):
        required_columns = [
            self.cfg.formula_column,
            self.cfg.sg_symbol_column,
            self.cfg.sg_number_column,
            self.cfg.natoms_column,
        ]

        if stage in (None, "fit"):
            train_df = pd.read_csv(self.cfg.train_csv)
            val_df = pd.read_csv(self.cfg.val_csv)

            train_df = train_df.dropna(subset=required_columns).reset_index(drop=True)
            val_df = val_df.dropna(subset=required_columns).reset_index(drop=True)

            self.train_dataset = SGDataset(
                formulas=train_df[self.cfg.formula_column].tolist(),
                n_atoms=train_df[self.cfg.natoms_column].tolist(),
                space_group_symbols=train_df[self.cfg.sg_symbol_column].tolist(),
                space_group_numbers=train_df[self.cfg.sg_number_column].tolist(),
                wyckoff_template_path=self.cfg.wyckoff_template_path,
                max_atoms=self.cfg.max_atoms,
            )
            self.val_dataset = SGDataset(
                formulas=val_df[self.cfg.formula_column].tolist(),
                n_atoms=val_df[self.cfg.natoms_column].tolist(),
                space_group_symbols=val_df[self.cfg.sg_symbol_column].tolist(),
                space_group_numbers=val_df[self.cfg.sg_number_column].tolist(),
                formula_vocab=self.train_dataset.formula_vocab,
                space_group_vocab=self.train_dataset.space_group_vocab,
                max_atoms=self.cfg.max_atoms,
            )

            self.formula_vocab = self.train_dataset.formula_vocab
            self.space_group_vocab = self.train_dataset.space_group_vocab

        if stage in (None, "test") and self.cfg.test_csv:
            test_df = pd.read_csv(self.cfg.test_csv)
            test_df = test_df.dropna(subset=required_columns).reset_index(drop=True)

            self.test_dataset = SGDataset(
                formulas=test_df[self.cfg.formula_column].tolist(),
                n_atoms=test_df[self.cfg.natoms_column].tolist(),
                space_group_symbols=test_df[self.cfg.sg_symbol_column].tolist(),
                space_group_numbers=test_df[self.cfg.sg_number_column].tolist(),
                formula_vocab=self.formula_vocab,
                space_group_vocab=self.space_group_vocab,
                max_atoms=self.cfg.max_atoms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last,
            collate_fn=sg_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=sg_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=sg_collate_fn,
        )
