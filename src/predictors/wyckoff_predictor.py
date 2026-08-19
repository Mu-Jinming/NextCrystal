# path: src/predictors/wyckoff_predictor.py
from __future__ import annotations

import json
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.wyckoff_predictor import WyckoffPredictor
from src.utils.inference_checkpoint import (
    get_checkpoint_vocab,
    load_checkpoint,
    load_model_state_dict,
)
from src.utils.wyckoff_template import (
    build_space_group_vocab,
    load_space_group_symbols,
)


ELEMENT_ORDER = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
ELEMENT_ORDER_DICT = {element: idx for idx, element in enumerate(ELEMENT_ORDER)}


@dataclass
class WyckoffPredictConfig:
    model_path: str
    wyckoff_template_path: str
    input_csv: str
    output_csv: str

    # Legacy five-file checkpoints remain readable, but new checkpoints bundle
    # all three vocabularies and leave these paths unset.
    formula_vocab_path: Optional[str] = None
    space_group_vocab_path: Optional[str] = None
    wyckoff_vocab_path: Optional[str] = None
    full_probabilities_npz: Optional[str] = None

    formula_column: str = "pretty_formula"
    natoms_column: str = "NAtoms"
    sg_symbol_column: str = "Spacegroup Symbol"
    sg_number_column: str = "Spacegroup Number"
    wyckoff_column: str = "Wyckoff Positions"

    top_n: int = 6
    max_atoms: int = 512
    max_multiplicities: int = 30
    batch_size: int = 64
    num_workers: int = 4
    use_data_parallel: bool = False

    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    dim_feedforward: int = 3072
    dropout: float = 0.1

    use_moe: bool = True

    moe_layers: str = "all"
    num_experts: int = 8
    capacity_factor: float = 1.5
    router_noisy_std: float = 0.5
    moe_loss_coef: float = 0.002
    moe_top_k: int = 2


def load_multiplicity_dict(template_path: str) -> dict[int, list[int]]:
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Wyckoff template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        data_template = json.load(f)

    multiplicity_dict: dict[int, list[int]] = {}
    for group in data_template["space_groups"]:
        number = int(group["number"])
        multiplicities = [int(pos["multiplicity"]) for pos in group["wyckoff_positions"]]
        if number in multiplicity_dict:
            multiplicity_dict[number].extend(multiplicities)
        else:
            multiplicity_dict[number] = multiplicities
    return multiplicity_dict


class WyckoffInferDataset(Dataset):
    def __init__(
        self,
        formulas,
        n_atoms,
        space_group_symbols,
        space_group_numbers,
        multiplicity_dict,
        formula_vocab,
        space_group_vocab,
        wyckoff_vocab,
        number_ordered_space_group_vocab: bool = False,
        max_atoms: int = 512,
        max_multiplicities: int = 30,
    ):
        self.formulas = formulas
        self.n_atoms = n_atoms
        self.space_group_symbols = space_group_symbols
        self.space_group_numbers = space_group_numbers
        self.max_atoms = max_atoms
        self.max_multiplicities = max_multiplicities
        self.multiplicity_dict = multiplicity_dict
        self.formula_vocab = formula_vocab
        self.space_group_vocab = space_group_vocab
        self.wyckoff_vocab = wyckoff_vocab
        self.number_ordered_space_group_vocab = number_ordered_space_group_vocab

        self.elements_per_atom = [
            self.get_elements_per_atom(fml, n)
            for fml, n in zip(self.formulas, self.n_atoms)
        ]

    def parse_formula(self, formula):
        def parse_segment(segment):
            counts = defaultdict(int)
            pattern = r"([A-Z][a-z]?)(\d*)|\(([^()]+)\)(\d*)"
            for m in re.finditer(pattern, segment):
                if m.group(1):
                    ele = m.group(1)
                    cnt = int(m.group(2)) if m.group(2) else 1
                    counts[ele] += cnt
                elif m.group(3):
                    inner = m.group(3)
                    mult = int(m.group(4)) if m.group(4) else 1
                    inner_counts = parse_segment(inner)
                    for ele, cnt in inner_counts.items():
                        counts[ele] += cnt * mult
            return counts

        counts = parse_segment(formula)
        total_atoms_in_fu = sum(counts.values())
        sorted_items = sorted(counts.items(), key=lambda kv: ELEMENT_ORDER_DICT.get(kv[0], float("inf")))
        return dict(OrderedDict(sorted_items)), total_atoms_in_fu

    def get_elements_per_atom(self, formula, n_atoms):
        elem_counts, total_fu = self.parse_formula(formula)
        elem_total = {}
        total_floor = 0
        fracs = []
        for ele, cnt in elem_counts.items():
            tot = n_atoms * cnt / total_fu
            fl = int(tot)
            frac = tot - fl
            elem_total[ele] = fl
            total_floor += fl
            fracs.append((frac, ele))
        missing = n_atoms - total_floor
        fracs.sort(reverse=True)
        for i in range(missing):
            if i < len(fracs):
                _, ele = fracs[i]
                elem_total[ele] += 1
        elements = []
        for ele, cnt in elem_total.items():
            elements.extend([ele] * cnt)
        if len(elements) < n_atoms:
            elements.extend(["<pad>"] * (n_atoms - len(elements)))
        elif len(elements) > n_atoms:
            elements = elements[:n_atoms]
        return elements

    def formula_to_tokens(self, formula, n_atoms):
        elem_counts, total_fu = self.parse_formula(formula)
        elem_total = {}
        total_floor = 0
        fracs = []
        for ele, cnt in elem_counts.items():
            tot = n_atoms * cnt / total_fu
            fl = int(tot)
            frac = tot - fl
            elem_total[ele] = fl
            total_floor += fl
            fracs.append((frac, ele))

        missing = n_atoms - total_floor
        fracs.sort(reverse=True)
        for i in range(missing):
            if i < len(fracs):
                _, ele = fracs[i]
                elem_total[ele] += 1

        tokens = []
        for ele, cnt in elem_total.items():
            idx = self.formula_vocab.get(ele, self.formula_vocab["<unk>"])
            tokens.extend([idx] * cnt)

        if len(tokens) > self.max_atoms:
            tokens = tokens[: self.max_atoms]
            mask = [1] * self.max_atoms
        else:
            mask = [1] * len(tokens) + [0] * (self.max_atoms - len(tokens))
            tokens += [self.formula_vocab["<pad>"]] * (self.max_atoms - len(tokens))

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def space_group_to_index(self, sg, sg_number):
        if self.number_ordered_space_group_vocab:
            return int(sg_number) - 1
        if sg in self.space_group_vocab:
            return self.space_group_vocab[sg]
        if "<unk>" in self.space_group_vocab:
            return self.space_group_vocab["<unk>"]
        raise ValueError(f"Space group {sg!r} (No. {sg_number}) is not in the vocabulary")

    def multiplicity_to_indices(self, sg_num):
        multiplicities = self.multiplicity_dict.get(sg_num, [])
        if len(multiplicities) > self.max_multiplicities:
            multiplicities = multiplicities[: self.max_multiplicities]
            mk = [1] * self.max_multiplicities
        else:
            mk = [1] * len(multiplicities) + [0] * (self.max_multiplicities - len(multiplicities))
            multiplicities += [0] * (self.max_multiplicities - len(multiplicities))
        return torch.tensor(multiplicities, dtype=torch.long), torch.tensor(mk, dtype=torch.bool)

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        formula = self.formulas[idx]
        n_atom = self.n_atoms[idx]
        sg_symbol = self.space_group_symbols[idx]
        sg_number = self.space_group_numbers[idx]

        tokens, mask = self.formula_to_tokens(formula, n_atom)
        sg_idx = self.space_group_to_index(sg_symbol, sg_number)
        multiplicities, multiplicity_mask = self.multiplicity_to_indices(sg_number)
        elements = self.elements_per_atom[idx]

        if len(elements) < self.max_atoms:
            elements += ["<pad>"] * (self.max_atoms - len(elements))
        else:
            elements = elements[: self.max_atoms]

        return tokens, mask, sg_idx, multiplicities, multiplicity_mask, elements


def wyckoff_infer_collate_fn(batch):
    tokens, masks, sgs, multiplicities, multiplicity_masks, elements = zip(*batch)
    return (
        torch.stack(tokens),
        torch.stack(masks),
        torch.tensor(sgs, dtype=torch.long),
        torch.stack(multiplicities),
        torch.stack(multiplicity_masks),
        list(elements),
    )


class WyckoffSequencePredictor:

    def __init__(self, cfg: WyckoffPredictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        checkpoint = load_checkpoint(
            cfg.model_path,
            map_location="cpu",
            expected_task="wyckoff",
        )
        self.formula_vocab = get_checkpoint_vocab(checkpoint, "formula_vocab")
        self.space_group_vocab = get_checkpoint_vocab(
            checkpoint, "space_group_vocab"
        )
        self.wyckoff_vocab = get_checkpoint_vocab(checkpoint, "wyckoff_vocab")

        if self.formula_vocab is None:
            needed = {
                "formula_vocab_path": cfg.formula_vocab_path,
                "space_group_vocab_path": cfg.space_group_vocab_path,
                "wyckoff_vocab_path": cfg.wyckoff_vocab_path,
            }
            missing = [
                name
                for name, path in needed.items()
                if not path or not os.path.isfile(path)
            ]
            if missing:
                raise FileNotFoundError(
                    "Legacy checkpoint requires existing vocabulary files: "
                    + ", ".join(missing)
                )
            self.formula_vocab = torch.load(
                cfg.formula_vocab_path,
                map_location="cpu",
                weights_only=False,
            )
            self.space_group_vocab = torch.load(
                cfg.space_group_vocab_path,
                map_location="cpu",
                weights_only=False,
            )
            self.wyckoff_vocab = torch.load(
                cfg.wyckoff_vocab_path,
                map_location="cpu",
                weights_only=False,
            )

        if self.space_group_vocab is None or self.wyckoff_vocab is None:
            raise ValueError("Checkpoint is missing a required vocabulary")
        self.wyckoff_idx_to_label = {
            idx: label for label, idx in self.wyckoff_vocab.items()
        }

        fixed_space_group_vocab = build_space_group_vocab(cfg.wyckoff_template_path)
        self.space_group_number_to_symbol = load_space_group_symbols(
            cfg.wyckoff_template_path
        )
        self.number_ordered_space_group_vocab = all(
            self.space_group_vocab.get(symbol) == index
            for symbol, index in fixed_space_group_vocab.items()
        )

        self.multiplicity_dict = load_multiplicity_dict(cfg.wyckoff_template_path)
        multiplicity_vocab_size = max(
            [max(self.multiplicity_dict[num], default=0) for num in self.multiplicity_dict]
        )

        self.model = WyckoffPredictor(
            vocab_size=len(self.formula_vocab),
            space_group_size=len(self.space_group_vocab),
            wyckoff_size=len(self.wyckoff_vocab),
            multiplicity_vocab_size=multiplicity_vocab_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            max_len=cfg.max_atoms,
            dropout=cfg.dropout,
            max_multiplicities=cfg.max_multiplicities,
            use_moe=cfg.use_moe,
            moe_layers=cfg.moe_layers,
            num_experts=cfg.num_experts,
            capacity_factor=cfg.capacity_factor,
            router_noisy_std=cfg.router_noisy_std,
            moe_loss_coef=cfg.moe_loss_coef,
            moe_top_k=cfg.moe_top_k,
            formula_vocab=self.formula_vocab,
            enable_sg_head=False,
        )

        load_model_state_dict(self.model, checkpoint)

        if cfg.use_data_parallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(device)
        self.model.eval()

    def predict_dataframe(self) -> pd.DataFrame:
        cfg = self.cfg

        if not os.path.exists(cfg.input_csv):
            raise FileNotFoundError(f"no csv: {cfg.input_csv}")

        df = pd.read_csv(cfg.input_csv)
        required_columns = [
            cfg.formula_column,
            cfg.sg_number_column,
            cfg.natoms_column,
        ]
        if not all(c in df.columns for c in required_columns):
            raise ValueError(f"no required_columns: {required_columns}")

        formulas = df[cfg.formula_column].tolist()
        n_atoms = df[cfg.natoms_column].astype(int).tolist()
        raw_sg_numbers = pd.to_numeric(df[cfg.sg_number_column], errors="raise")
        if not np.equal(raw_sg_numbers, np.floor(raw_sg_numbers)).all():
            raise ValueError("Spacegroup Number values must be integers in [1, 230]")
        sg_numbers = raw_sg_numbers.astype(int).tolist()
        invalid_sg_numbers = sorted(
            number
            for number in set(sg_numbers)
            if number not in self.space_group_number_to_symbol
        )
        if invalid_sg_numbers:
            raise ValueError(
                "Spacegroup Number values must be in [1, 230]; "
                f"invalid={invalid_sg_numbers}"
            )
        sg_symbols = [
            self.space_group_number_to_symbol[number] for number in sg_numbers
        ]
        df = df.copy()
        df[cfg.sg_symbol_column] = sg_symbols

        dataset = WyckoffInferDataset(
            formulas=formulas,
            n_atoms=n_atoms,
            space_group_symbols=sg_symbols,
            space_group_numbers=sg_numbers,
            multiplicity_dict=self.multiplicity_dict,
            formula_vocab=self.formula_vocab,
            space_group_vocab=self.space_group_vocab,
            wyckoff_vocab=self.wyckoff_vocab,
            number_ordered_space_group_vocab=self.number_ordered_space_group_vocab,
            max_atoms=cfg.max_atoms,
            max_multiplicities=cfg.max_multiplicities,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=wyckoff_infer_collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        predictions = []
        topk_predictions = []
        save_full_probabilities = bool(cfg.full_probabilities_npz)
        full_probability_chunks = []
        full_elements = []
        full_row_offsets = [0]

        with torch.no_grad():
            for batch in dataloader:
                tokens, masks, sgs, multiplicities, multiplicity_masks, elements = batch
                tokens = tokens.to(self.device)
                masks = masks.to(self.device)
                sgs = sgs.to(self.device)
                multiplicities = multiplicities.to(self.device)
                multiplicity_masks = multiplicity_masks.to(self.device)

                out = self.model(tokens, masks, sgs, multiplicities, multiplicity_masks)
                logits = out[0] if isinstance(out, tuple) else out

                probs = torch.softmax(logits, dim=-1)
                _, top1_preds = probs.max(dim=-1)
                topk_probs, topk_preds = probs.topk(cfg.top_n, dim=-1)

                probs_np = probs.cpu().numpy()
                top1_preds = top1_preds.cpu().numpy()
                topk_probs = topk_probs.cpu().numpy()
                topk_preds = topk_preds.cpu().numpy()
                masks_np = masks.cpu().numpy()
                elements_list = list(elements)

                B, L = tokens.size()
                for i in range(B):
                    atom_count = int(masks_np[i].sum())
                    ele_list = elements_list[i][:atom_count]

                    atoms_top1 = top1_preds[i][:atom_count]
                    elem_letter_dict = [
                        {ele: self.wyckoff_idx_to_label.get(int(pred), "<unk>")}
                        for ele, pred in zip(ele_list, atoms_top1)
                    ]

                    atoms_topk = []
                    for j in range(atom_count):
                        per_atom = {}
                        for k in range(cfg.top_n):
                            label = self.wyckoff_idx_to_label.get(int(topk_preds[i][j][k]), "<unk>")
                            prob = float(topk_probs[i][j][k])
                            per_atom[label] = prob
                        atoms_topk.append(per_atom)

                    topk_elem_letter = [{ele: per_atom} for ele, per_atom in zip(ele_list, atoms_topk)]

                    predictions.append(json.dumps(elem_letter_dict, ensure_ascii=False))
                    topk_predictions.append(json.dumps(topk_elem_letter, ensure_ascii=False))
                    if save_full_probabilities:
                        full_probability_chunks.append(
                            np.asarray(probs_np[i, :atom_count], dtype=np.float32)
                        )
                        full_elements.extend(str(element) for element in ele_list)
                        full_row_offsets.append(full_row_offsets[-1] + atom_count)

        if save_full_probabilities:
            probability_path = Path(cfg.full_probabilities_npz)
            probability_path.parent.mkdir(parents=True, exist_ok=True)
            labels = [
                self.wyckoff_idx_to_label[index]
                for index in range(len(self.wyckoff_idx_to_label))
            ]
            if full_probability_chunks:
                probabilities = np.concatenate(full_probability_chunks, axis=0)
            else:
                probabilities = np.empty((0, len(labels)), dtype=np.float32)
            np.savez_compressed(
                probability_path,
                probabilities=probabilities,
                row_offsets=np.asarray(full_row_offsets, dtype=np.int64),
                elements=np.asarray(full_elements, dtype="<U3"),
                labels=np.asarray(labels, dtype="<U16"),
                space_group_numbers=np.asarray(sg_numbers, dtype=np.int16),
            )

        out_rows = []
        has_cif_name = "cif_name" in df.columns
        has_sg_prob = "SG_Prob" in df.columns
        has_sg_rank = "SG_Rank" in df.columns

        for idx in range(len(df)):
            row = {
                "cif_name": df.loc[idx, "cif_name"] if has_cif_name else None,
                "Formula pretty" if cfg.formula_column == "pretty_formula" else "Formula": df.loc[idx, cfg.formula_column],
                "Spacegroup Symbol": df.loc[idx, cfg.sg_symbol_column],
                "Spacegroup Number": int(df.loc[idx, cfg.sg_number_column]),
                "NAtoms": int(df.loc[idx, cfg.natoms_column]),
                "Predicted Wyckoff Letters": predictions[idx],
                f"Top-{cfg.top_n} Predicted Wyckoff Letters": topk_predictions[idx],
            }
            if has_sg_prob:
                row["SG_Prob"] = float(df.loc[idx, "SG_Prob"])
            if has_sg_rank:
                row["SG_Rank"] = int(df.loc[idx, "SG_Rank"])
            out_rows.append(row)

        return pd.DataFrame(out_rows)

    def save_predictions(self) -> Path:
        output_path = Path(self.cfg.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df = self.predict_dataframe()
        result_df.to_csv(output_path, index=False)
        return output_path
