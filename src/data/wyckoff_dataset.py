# path: src/data/wyckoff_dataset.py
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
class WyckoffDataConfig:
    train_csv: str
    val_csv: str
    test_csv: Optional[str] = None
    wyckoff_template_path: str = "data/wyckoff_template.json"

    formula_column: str = "formula"
    natoms_column: str = "NAtoms"
    sg_symbol_column: str = "Spacegroup Symbol"
    sg_number_column: str = "Spacegroup Number"
    wyckoff_column: str = "Wyckoff Positions"

    max_atoms: int = 512
    max_multiplicities: int = 30
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False


class WyckoffDataset(Dataset):
    def __init__(
        self,
        formulas,
        n_atoms,
        space_group_symbols,
        space_group_numbers,
        wyckoff_positions,
        multiplicity_dict,
        formula_vocab=None,
        space_group_vocab=None,
        wyckoff_vocab=None,
        wyckoff_template_path: str = "data/wyckoff_template.json",
        max_atoms: int = 512,
        max_multiplicities: int = 30,
    ):
        self.formulas = list(formulas)
        self.n_atoms = [int(x) for x in n_atoms]
        self.space_group_symbols = list(space_group_symbols)
        self.space_group_numbers = [int(x) for x in space_group_numbers]
        self.wyckoff_positions = list(wyckoff_positions)
        self.max_atoms = max_atoms
        self.max_multiplicities = max_multiplicities
        self.multiplicity_dict = multiplicity_dict
        self.wyckoff_template_path = wyckoff_template_path

        self.formula_vocab = formula_vocab or build_formula_vocab(self.formulas)
        self.space_group_vocab = (
            space_group_vocab
            if space_group_vocab is not None
            else self.build_space_group_vocab()
        )
        self.wyckoff_vocab = wyckoff_vocab or self.build_wyckoff_vocab()

    def build_space_group_vocab(self):
        return build_space_group_vocab(self.wyckoff_template_path)

    def build_wyckoff_vocab(self):
        unique_wyckoffs = set()
        for wyckoff_str in self.wyckoff_positions:
            if not isinstance(wyckoff_str, str):
                continue
            wyckoff_letters = [item.split(":")[1].strip() for item in wyckoff_str.split(",") if ":" in item]
            unique_wyckoffs.update(wyckoff_letters)
        wyckoff_list = sorted(list(unique_wyckoffs))
        wyckoff_to_idx = {letter: idx for idx, letter in enumerate(wyckoff_list)}
        wyckoff_to_idx["<pad>"] = len(wyckoff_to_idx)
        return wyckoff_to_idx

    def wyckoff_to_indices(self, wyckoff_str):
        wyckoff_letters = [item.split(":")[1].strip() for item in wyckoff_str.split(",") if ":" in item]
        wyckoff_indices = [self.wyckoff_vocab.get(letter, self.wyckoff_vocab["<pad>"]) for letter in wyckoff_letters]
        return wyckoff_indices

    def space_group_to_index(self, space_group_number):
        return space_group_number_to_index(space_group_number)

    def multiplicity_to_indices(self, space_group_number):
        multiplicities = list(self.multiplicity_dict.get(space_group_number, []))
        if len(multiplicities) > self.max_multiplicities:
            multiplicities = multiplicities[:self.max_multiplicities]
            mask = [1] * self.max_multiplicities
        else:
            mask = [1] * len(multiplicities) + [0] * (self.max_multiplicities - len(multiplicities))
            multiplicities += [0] * (self.max_multiplicities - len(multiplicities))
        return (
            torch.tensor(multiplicities, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        tokens, mask = formula_to_tokens(
            formula=self.formulas[idx],
            n_atoms=self.n_atoms[idx],
            formula_vocab=self.formula_vocab,
            max_atoms=self.max_atoms,
        )

        wyckoff_indices = self.wyckoff_to_indices(self.wyckoff_positions[idx])
        if len(wyckoff_indices) > self.max_atoms:
            wyckoff_indices = wyckoff_indices[: self.max_atoms]
        else:
            wyckoff_indices += [self.wyckoff_vocab["<pad>"]] * (self.max_atoms - len(wyckoff_indices))

        multiplicities, multiplicity_mask = self.multiplicity_to_indices(self.space_group_numbers[idx])
        space_group_idx = self.space_group_to_index(self.space_group_numbers[idx])

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "labels": torch.tensor(wyckoff_indices, dtype=torch.long),
            "space_group": torch.tensor(space_group_idx, dtype=torch.long),
            "multiplicities": multiplicities,
            "multiplicity_mask": multiplicity_mask,
        }


def wyckoff_collate_fn(batch):
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "space_group": torch.stack([item["space_group"] for item in batch]),
        "multiplicities": torch.stack([item["multiplicities"] for item in batch]),
        "multiplicity_mask": torch.stack([item["multiplicity_mask"] for item in batch]),
    }


class WyckoffDataModule(L.LightningDataModule):
    def __init__(self, cfg: WyckoffDataConfig, multiplicity_dict: dict[int, list[int]]):
        super().__init__()
        self.cfg = cfg
        self.multiplicity_dict = multiplicity_dict

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.formula_vocab = None
        self.space_group_vocab = None
        self.wyckoff_vocab = None

    def setup(self, stage: Optional[str] = None):
        required_columns = [
            self.cfg.formula_column,
            self.cfg.sg_symbol_column,
            self.cfg.sg_number_column,
            self.cfg.natoms_column,
            self.cfg.wyckoff_column,
        ]

        if stage in (None, "fit"):
            train_df = pd.read_csv(self.cfg.train_csv)
            val_df = pd.read_csv(self.cfg.val_csv)

            train_df = train_df.dropna(subset=required_columns).reset_index(drop=True)
            val_df = val_df.dropna(subset=required_columns).reset_index(drop=True)

            self.train_dataset = WyckoffDataset(
                formulas=train_df[self.cfg.formula_column].tolist(),
                n_atoms=train_df[self.cfg.natoms_column].tolist(),
                space_group_symbols=train_df[self.cfg.sg_symbol_column].tolist(),
                space_group_numbers=train_df[self.cfg.sg_number_column].tolist(),
                wyckoff_positions=train_df[self.cfg.wyckoff_column].tolist(),
                multiplicity_dict=self.multiplicity_dict,
                wyckoff_template_path=self.cfg.wyckoff_template_path,
                max_atoms=self.cfg.max_atoms,
                max_multiplicities=self.cfg.max_multiplicities,
            )
            self.val_dataset = WyckoffDataset(
                formulas=val_df[self.cfg.formula_column].tolist(),
                n_atoms=val_df[self.cfg.natoms_column].tolist(),
                space_group_symbols=val_df[self.cfg.sg_symbol_column].tolist(),
                space_group_numbers=val_df[self.cfg.sg_number_column].tolist(),
                wyckoff_positions=val_df[self.cfg.wyckoff_column].tolist(),
                multiplicity_dict=self.multiplicity_dict,
                formula_vocab=self.train_dataset.formula_vocab,
                space_group_vocab=self.train_dataset.space_group_vocab,
                wyckoff_vocab=self.train_dataset.wyckoff_vocab,
                max_atoms=self.cfg.max_atoms,
                max_multiplicities=self.cfg.max_multiplicities,
            )

            self.formula_vocab = self.train_dataset.formula_vocab
            self.space_group_vocab = self.train_dataset.space_group_vocab
            self.wyckoff_vocab = self.train_dataset.wyckoff_vocab

        if stage in (None, "test") and self.cfg.test_csv:
            test_df = pd.read_csv(self.cfg.test_csv)
            test_df = test_df.dropna(subset=required_columns).reset_index(drop=True)

            self.test_dataset = WyckoffDataset(
                formulas=test_df[self.cfg.formula_column].tolist(),
                n_atoms=test_df[self.cfg.natoms_column].tolist(),
                space_group_symbols=test_df[self.cfg.sg_symbol_column].tolist(),
                space_group_numbers=test_df[self.cfg.sg_number_column].tolist(),
                wyckoff_positions=test_df[self.cfg.wyckoff_column].tolist(),
                multiplicity_dict=self.multiplicity_dict,
                formula_vocab=self.formula_vocab,
                space_group_vocab=self.space_group_vocab,
                wyckoff_vocab=self.wyckoff_vocab,
                max_atoms=self.cfg.max_atoms,
                max_multiplicities=self.cfg.max_multiplicities,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last,
            collate_fn=wyckoff_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=wyckoff_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=wyckoff_collate_fn,
        )
