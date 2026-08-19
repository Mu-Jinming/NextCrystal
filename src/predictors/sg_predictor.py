# path: src/predictors/sg_predictor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from src.models.wyckoff_predictor import WyckoffPredictor
from src.utils.chemistry import build_formula_vocab, formula_to_tokens
from src.utils.inference_checkpoint import (
    extract_model_state_dict,
    get_checkpoint_metadata,
    get_checkpoint_vocab,
    load_checkpoint,
    load_model_state_dict,
)
from src.utils.wyckoff_template import (
    NUM_SPACE_GROUPS,
    build_space_group_vocab,
    load_space_group_symbols,
)


@dataclass
class SGPredictConfig:
    model_path: str
    input_csv: str
    output_csv: str

    train_csv: Optional[str] = None
    space_group_vocab_path: Optional[str] = None
    val_csv: Optional[str] = None
    test_csv: Optional[str] = None
    wyckoff_template_path: str = "data/wyckoff_template.json"

    formula_column: str = "formula"
    natoms_column: str = "NAtoms"
    sg_symbol_column: str = "Spacegroup Symbol"
    sg_number_column: str = "Spacegroup Number"

    max_atoms: int = 512
    batch_size: int = 256
    top_k: int = 5

    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1

    use_moe: bool = True
    moe_layers: str | list[int] = "all"
    num_experts: int = 8
    capacity_factor: float = 1.5
    router_noisy_std: float = 0.5
    moe_loss_coef: float = 0.002
    moe_top_k: int = 2


def _build_observed_space_group_vocab(space_group_symbols):
    """Reconstruct the label space used by legacy checkpoints."""
    unique_space_groups = sorted(set(space_group_symbols))
    space_group_to_idx = {sg: idx for idx, sg in enumerate(unique_space_groups)}
    space_group_to_idx["<unk>"] = len(space_group_to_idx)
    return space_group_to_idx


def _infer_checkpoint_space_group_size(state) -> int:
    classifier_weights = [
        value
        for key, value in state.items()
        if key.endswith("sg_classifier.weight")
    ]
    if len(classifier_weights) != 1:
        raise ValueError(
            "Expected one sg_classifier.weight in the checkpoint, "
            f"found {len(classifier_weights)}"
        )
    return int(classifier_weights[0].shape[0])


def _read_existing_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_vocabs_and_mappings(
    cfg: SGPredictConfig,
    space_group_size: int,
    *,
    checkpoint_formula_vocab: Optional[dict[str, int]] = None,
    checkpoint_space_group_vocab: Optional[dict[str, int]] = None,
    checkpoint_symbol_to_number: Optional[dict[str, int]] = None,
):
    frames = []

    train_df = _read_existing_csv(cfg.train_csv)
    if train_df is None and (
        checkpoint_formula_vocab is None
        or checkpoint_space_group_vocab is None
    ):
        raise FileNotFoundError(f"train_csv not found: {cfg.train_csv}")
    if train_df is not None:
        frames.append(train_df)

    val_df = _read_existing_csv(cfg.val_csv)
    if val_df is not None:
        frames.append(val_df)

    test_df = _read_existing_csv(cfg.test_csv)
    if test_df is not None:
        frames.append(test_df)

    if frames:
        df = pd.concat(frames, axis=0, ignore_index=True)
        required = [
            cfg.formula_column,
            cfg.sg_symbol_column,
            cfg.sg_number_column,
            cfg.natoms_column,
        ]
        if df[required].isnull().any().any():
            df = df.dropna(subset=required)
        formulas = df[cfg.formula_column].tolist()
        sg_symbols = df[cfg.sg_symbol_column].tolist()
        sg_numbers = df[cfg.sg_number_column].tolist()
    else:
        formulas = []
        sg_symbols = []
        sg_numbers = []

    formula_vocab = (
        checkpoint_formula_vocab
        if checkpoint_formula_vocab is not None
        else build_formula_vocab(formulas)
    )
    number_to_symbol = load_space_group_symbols(cfg.wyckoff_template_path)
    observed_symbol_to_number = {
        str(symbol): int(number)
        for symbol, number in zip(sg_symbols, sg_numbers)
    }
    canonical_symbol_to_number = {
        symbol: number for number, symbol in number_to_symbol.items()
    }
    sg_symbol_to_number = (
        canonical_symbol_to_number
        | dict(checkpoint_symbol_to_number or {})
        | observed_symbol_to_number
    )

    vocab_path = (
        Path(cfg.space_group_vocab_path) if cfg.space_group_vocab_path else None
    )
    if checkpoint_space_group_vocab is not None:
        space_group_vocab = checkpoint_space_group_vocab
        vocab_source = "the bundled space-group vocabulary"
        if len(space_group_vocab) != space_group_size:
            raise ValueError(
                f"Checkpoint has {space_group_size} classes but {vocab_source} "
                f"contains {len(space_group_vocab)} tokens"
            )
        if set(space_group_vocab.values()) != set(range(space_group_size)):
            raise ValueError(
                f"Space-group vocabulary indices must be dense: {vocab_source}"
            )
    elif vocab_path is not None:
        if not vocab_path.exists():
            raise FileNotFoundError(f"space_group_vocab_path not found: {vocab_path}")
        space_group_vocab = torch.load(
            vocab_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(space_group_vocab, dict):
            raise ValueError(f"Space-group vocabulary must be a dict: {vocab_path}")
        if len(space_group_vocab) != space_group_size:
            raise ValueError(
                f"Checkpoint has {space_group_size} classes but {vocab_path} "
                f"contains {len(space_group_vocab)} tokens"
            )
        if set(space_group_vocab.values()) != set(range(space_group_size)):
            raise ValueError(f"Space-group vocabulary indices must be dense: {vocab_path}")
    elif space_group_size in (NUM_SPACE_GROUPS, NUM_SPACE_GROUPS + 1):
        include_unknown = space_group_size == NUM_SPACE_GROUPS + 1
        space_group_vocab = build_space_group_vocab(
            cfg.wyckoff_template_path,
            include_unknown=include_unknown,
        )
    else:
        # Keep existing, already-trained checkpoints usable.  New training
        # always constructs the fixed 230-class vocabulary in the data module.
        space_group_vocab = _build_observed_space_group_vocab(sg_symbols)
        if len(space_group_vocab) != space_group_size:
            raise ValueError(
                "Checkpoint/output vocabulary mismatch: "
                f"checkpoint has {space_group_size} classes but the legacy "
                f"data-derived vocabulary has {len(space_group_vocab)}"
            )

    return formula_vocab, space_group_vocab, sg_symbol_to_number


class SpaceGroupPredictor:

    def __init__(self, cfg: SGPredictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        checkpoint = load_checkpoint(
            cfg.model_path,
            map_location="cpu",
            expected_task="spacegroup",
        )
        state_dict = extract_model_state_dict(checkpoint)
        space_group_size = _infer_checkpoint_space_group_size(state_dict)
        checkpoint_metadata = get_checkpoint_metadata(checkpoint)
        (
            self.formula_vocab,
            self.space_group_vocab,
            self.sg_symbol_to_number,
        ) = load_vocabs_and_mappings(
            cfg,
            space_group_size=space_group_size,
            checkpoint_formula_vocab=get_checkpoint_vocab(
                checkpoint, "formula_vocab"
            ),
            checkpoint_space_group_vocab=get_checkpoint_vocab(
                checkpoint, "space_group_vocab"
            ),
            checkpoint_symbol_to_number=checkpoint_metadata.get(
                "space_group_symbol_to_number"
            ),
        )
        self.idx_to_sg_symbol = {
            idx: sg for sg, idx in self.space_group_vocab.items() if sg != "<unk>"
        }
        self.unk_idx = self.space_group_vocab.get("<unk>", None)

        self.model = WyckoffPredictor(
            vocab_size=len(self.formula_vocab),
            space_group_size=len(self.space_group_vocab),
            wyckoff_size=2,
            multiplicity_vocab_size=1,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            max_len=cfg.max_atoms,
            dropout=cfg.dropout,
            max_multiplicities=1,
            use_moe=cfg.use_moe,
            moe_layers=cfg.moe_layers,
            num_experts=cfg.num_experts,
            capacity_factor=cfg.capacity_factor,
            router_noisy_std=cfg.router_noisy_std,
            moe_loss_coef=cfg.moe_loss_coef,
            moe_top_k=cfg.moe_top_k,
            formula_vocab=self.formula_vocab,
            enable_sg_head=True,
        ).to(device)

        load_model_state_dict(self.model, checkpoint)
        self.model.eval()

    def predict_dataframe(self) -> pd.DataFrame:
        cfg = self.cfg
        df = pd.read_csv(cfg.input_csv)
        df.columns = df.columns.str.strip()

        required = [cfg.formula_column, cfg.natoms_column]
        if not all(c in df.columns for c in required):
            raise ValueError(f"Input CSV must contain columns: {required}")

        if "cif_name" in df.columns:
            sample_names = df["cif_name"].astype(str).tolist()
        elif "material_id" in df.columns:
            sample_names = df["material_id"].astype(str).tolist()
        elif "id" in df.columns:
            sample_names = df["id"].astype(str).tolist()
        else:
            sample_names = df.index.astype(str).tolist()

        valid_mask = ~df[[cfg.formula_column, cfg.natoms_column]].isnull().any(axis=1)
        valid_indices = df.index[valid_mask].tolist()
        if len(valid_indices) == 0:
            raise ValueError("No valid rows found for prediction.")

        rows = []

        with torch.no_grad():
            for start in range(0, len(valid_indices), cfg.batch_size):
                batch_idx = valid_indices[start : start + cfg.batch_size]

                batch_names = [sample_names[i] for i in batch_idx]
                batch_formulas = [df.at[i, cfg.formula_column] for i in batch_idx]
                batch_natoms = [int(df.at[i, cfg.natoms_column]) for i in batch_idx]

                tokens_batch = []
                masks_batch = []
                for formula, n_atoms in zip(batch_formulas, batch_natoms):
                    tokens, mask = formula_to_tokens(
                        formula=formula,
                        n_atoms=n_atoms,
                        formula_vocab=self.formula_vocab,
                        max_atoms=cfg.max_atoms,
                    )
                    tokens_batch.append(torch.tensor(tokens, dtype=torch.long))
                    masks_batch.append(torch.tensor(mask, dtype=torch.bool))

                tokens_tensor = torch.stack(tokens_batch).to(self.device)
                masks_tensor = torch.stack(masks_batch).to(self.device)

                logits, _ = self.model.forward_sg(tokens_tensor, masks_tensor)
                probs = torch.softmax(logits, dim=-1)
                topk_probs, topk_indices = probs.topk(cfg.top_k, dim=-1)

                topk_probs = topk_probs.cpu().numpy()
                topk_indices = topk_indices.cpu().numpy()

                for sample_name, formula, n_atoms, idxs, probs_k in zip(
                    batch_names, batch_formulas, batch_natoms, topk_indices, topk_probs
                ):
                    for rank, (idx_k, prob_k) in enumerate(zip(idxs, probs_k), start=1):
                        idx_k = int(idx_k)
                        prob_k = float(prob_k)

                        if idx_k == self.unk_idx:
                            sym_k = "<unk>"
                            num_k = -1
                        else:
                            sym_k = self.idx_to_sg_symbol.get(idx_k, "<unk>")
                            num_k = self.sg_symbol_to_number.get(sym_k, -1)

                        rows.append(
                            {
                                "cif_name": str(sample_name),
                                cfg.formula_column: str(formula),
                                "NAtoms": int(n_atoms),
                                "Spacegroup Symbol": str(sym_k),
                                "Spacegroup Number": int(num_k),
                                "SG_Prob": float(prob_k),
                                "SG_Rank": int(rank),
                            }
                        )

        return pd.DataFrame(rows)

    def save_predictions(self) -> Path:
        output_path = Path(self.cfg.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.predict_dataframe()
        df.to_csv(output_path, index=False, encoding="utf-8")
        return output_path
