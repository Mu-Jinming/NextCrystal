"""Minimal NextDiff checkpoint and tensor helpers used by sampling."""

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir


def lattices_to_params_shape(lattices):
    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        cosine = torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1)
        cosine = cosine / (lengths[..., j] * lengths[..., k])
        angles[..., i] = torch.clamp(cosine, -1.0, 1.0)
    angles = torch.arccos(angles) * 180.0 / np.pi
    return lengths, angles


def load_model(model_path, load_data=False, testing=True):
    """Load the sampling model without instantiating any dataset.

    ``load_data`` and ``testing`` are retained for API compatibility with the
    upstream helper. The minimal release intentionally excludes training data
    and therefore rejects data-loader construction.
    """
    if load_data:
        raise ValueError("The minimal sampling runtime does not bundle datasets.")

    model_path = Path(model_path).expanduser().resolve()
    with initialize_config_dir(config_dir=str(model_path), version_base=None):
        cfg = compose(config_name="hparams")
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )

    checkpoints = sorted(model_path.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt file found under {model_path}")
    last_checkpoints = [path for path in checkpoints if "last" in path.name]
    checkpoint = last_checkpoints[-1] if last_checkpoints else checkpoints[-1]

    model = model.__class__.load_from_checkpoint(
        str(checkpoint),
        hparams_file=str(model_path / "hparams.yaml"),
        strict=True,
        map_location="cpu",
    )

    lattice_scaler = model_path / "lattice_scaler.pt"
    property_scaler = model_path / "prop_scaler.pt"
    if lattice_scaler.exists():
        model.lattice_scaler = torch.load(lattice_scaler, map_location="cpu")
    if property_scaler.exists():
        model.scaler = torch.load(property_scaler, map_location="cpu")

    return model, None, cfg


def get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms):
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, atom_count in enumerate(num_atoms.tolist()):
        crystal_array_list.append(
            {
                "frac_coords": frac_coords.narrow(0, start_idx, atom_count)
                .detach()
                .cpu()
                .numpy(),
                "atom_types": atom_types.narrow(0, start_idx, atom_count)
                .detach()
                .cpu()
                .numpy(),
                "lengths": lengths[batch_idx].detach().cpu().numpy(),
                "angles": angles[batch_idx].detach().cpu().numpy(),
            }
        )
        start_idx += atom_count
    return crystal_array_list
