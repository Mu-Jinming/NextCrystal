"""Symmetry-conditioned diffusion module used by NextDiff."""

import math
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm

from nextdiff.common.data_utils import lattice_params_to_matrix_torch
from nextdiff.pl_modules.lattice.crystal_family import CrystalFamily
from nextdiff.pl_modules.diff_utils import d_log_p_wrapped_normal


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # Populate self.hparams from the constructor arguments.
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


# Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """Sinusoidal embedding of the diffusion time step."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=time.device) * -scale
        )
        phases = time[:, None] * frequencies[None, :]
        return torch.cat((phases.sin(), phases.cos()), dim=-1)


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            latent_dim=self.hparams.time_dim,
            _recursive_=False,
        )
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.crystal_family = CrystalFamily()

    @staticmethod
    def _expand_orbit_vectors(vectors, batch, inverse_anchor=True):
        """Map one anchor vector to every symmetry-equivalent site."""
        anchor_vectors = vectors[batch.anchor_index]
        anchor_ops = (
            batch.ops_inv[batch.anchor_index]
            if inverse_anchor
            else batch.ops[batch.anchor_index, :3, :3]
        )
        anchor_vectors = (
            anchor_ops @ anchor_vectors.unsqueeze(-1)
        ).squeeze(-1)
        expanded = (
            batch.ops[:, :3, :3] @ anchor_vectors.unsqueeze(-1)
        ).squeeze(-1)
        return expanded, anchor_vectors

    @staticmethod
    def _expand_anchor_coordinates(coordinates, batch):
        homogeneous = torch.cat(
            (
                coordinates[batch.anchor_index],
                torch.ones(batch.ops.size(0), 1).to(coordinates.device),
            ),
            dim=-1,
        ).unsqueeze(-1)
        return (batch.ops @ homogeneous).squeeze(-1)[:, :3] % 1.0

    @staticmethod
    def _rectify_vector_field(vectors, batch):
        projected = torch.einsum("bij, bj-> bi", batch.ops_inv, vectors)
        anchors = scatter(
            projected,
            batch.anchor_index,
            dim=0,
            reduce="mean",
        )[batch.anchor_index]
        return (
            batch.ops[:, :3, :3] @ anchors.unsqueeze(-1)
        ).squeeze(-1)

    def forward(self, batch, batch_idx=None):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattices = self.crystal_family.de_so3(lattices)
        frac_coords = batch.frac_coords

        rand_x = torch.randn_like(frac_coords)

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[
            :, None
        ]

        rand_x, rand_x_anchor = self._expand_orbit_vectors(rand_x, batch)
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0

        ori_crys_fam = self.crystal_family.m2v(lattices)
        ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            ori_crys_fam, batch.spacegroup
        )
        rand_crys_fam = torch.randn_like(ori_crys_fam)
        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            rand_crys_fam, batch.spacegroup
        )
        input_crys_fam = c0[:, None] * ori_crys_fam + c1[:, None] * rand_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            input_crys_fam, batch.spacegroup
        )

        pred_crys_fam, pred_x = self.decoder(
            time_emb,
            batch.atom_types,
            input_frac_coords,
            input_crys_fam,
            batch.num_atoms,
            batch.batch,
        )
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            pred_crys_fam, batch.spacegroup
        )

        pred_x_proj = torch.einsum("bij, bj-> bi", batch.ops_inv, pred_x)

        tar_x_anchor = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x_anchor, sigmas_per_atom
        ) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_crys_fam, rand_crys_fam)

        loss_coord = F.mse_loss(pred_x_proj, tar_x_anchor)

        loss = (
            self.hparams.cost_lattice * loss_lattice
            + self.hparams.cost_coord * loss_coord
        )

        return {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
        }

    @torch.no_grad()
    def sample(
        self,
        batch,
        diff_ratio=1.0,
        step_lr=1e-5,
        rectification=True,
        show_progress=True,
        return_trajectory=True,
    ):
        batch_size = batch.num_graphs

        x_T = torch.rand((batch.num_nodes, 3)).to(self.device)
        crys_fam_T = torch.randn((batch_size, 6)).to(self.device)
        if rectification:
            crys_fam_T = self.crystal_family.proj_k_to_spacegroup(
                crys_fam_T, batch.spacegroup
            )
        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            lattices = self.crystal_family.de_so3(lattices)
            ori_crys_fam = self.crystal_family.m2v(lattices)
            if rectification:
                ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(
                    ori_crys_fam, batch.spacegroup
                )

            frac_coords = batch.frac_coords

            rand_crys_fam, rand_x = (
                torch.randn_like(ori_crys_fam),
                torch.randn_like(frac_coords),
            )

            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)

            sigmas = self.sigma_scheduler.sigmas[time_start]

            if rectification:
                rand_x, _ = self._expand_orbit_vectors(
                    rand_x,
                    batch,
                    inverse_anchor=False,
                )

            crys_fam_T = c0 * ori_crys_fam + c1 * rand_crys_fam
            x_T = (frac_coords + sigmas * rand_x) % 1.0

        else:
            time_start = self.beta_scheduler.timesteps - 1

        l_T = self.crystal_family.v2m(crys_fam_T)

        if rectification:
            x_T = self._expand_anchor_coordinates(x_T, batch)
        else:
            x_T = x_T % 1.0

        traj = {
            time_start: {
                "num_atoms": batch.num_atoms,
                "atom_types": batch.atom_types,
                "frac_coords": x_T % 1.0,
                "lattices": l_T,
                "crys_fam": crys_fam_T,
            }
        }

        for t in tqdm(range(time_start, 0, -1), disable=not show_progress):
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]["frac_coords"]
            crys_fam_t = traj[t]["crys_fam"]

            # Corrector

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr / (
                sigma_norm * self.sigma_scheduler.sigma_begin**2
            )
            std_x = torch.sqrt(2 * step_size)

            if rectification:
                rand_x, _ = self._expand_orbit_vectors(rand_x, batch)

            pred_crys_fam, pred_x = self.decoder(
                time_emb,
                batch.atom_types,
                x_t,
                crys_fam_t,
                batch.num_atoms,
                batch.batch,
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            if rectification:
                pred_x = self._rectify_vector_field(pred_x, batch)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            crys_fam_t_minus_05 = crys_fam_t

            if rectification:
                x_t_minus_05 = self._expand_anchor_coordinates(
                    x_t_minus_05,
                    batch,
                )
            else:
                x_t_minus_05 = x_t_minus_05 % 1.0

            # Predictor

            rand_crys_fam = torch.randn_like(crys_fam_T)
            if rectification:
                rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(
                    rand_crys_fam, batch.spacegroup
                )
            ori_crys_fam = crys_fam_t
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = torch.sqrt(
                adjacent_sigma_x**2
                * (sigma_x**2 - adjacent_sigma_x**2)
                / sigma_x**2
            )

            if rectification:
                rand_x, _ = self._expand_orbit_vectors(rand_x, batch)

            pred_crys_fam, pred_x = self.decoder(
                time_emb,
                batch.atom_types,
                x_t_minus_05,
                crys_fam_t,
                batch.num_atoms,
                batch.batch,
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            crys_fam_t_minus_1 = (
                c0 * (ori_crys_fam - c1 * pred_crys_fam)
                + sigmas * rand_crys_fam
            )
            if rectification:
                crys_fam_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(
                    crys_fam_t_minus_1, batch.spacegroup
                )

            if rectification:
                pred_x = self._rectify_vector_field(pred_x, batch)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = self.crystal_family.v2m(crys_fam_t_minus_1)

            if rectification:
                x_t_minus_1 = self._expand_anchor_coordinates(
                    x_t_minus_1,
                    batch,
                )
            else:
                x_t_minus_1 = x_t_minus_1 % 1.0

            traj[t - 1] = {
                "num_atoms": batch.num_atoms,
                "atom_types": batch.atom_types,
                "frac_coords": x_t_minus_1 % 1.0,
                "lattices": l_t_minus_1,
                "crys_fam": crys_fam_t_minus_1,
            }

            if not return_trajectory:
                del traj[t]

        if return_trajectory:
            traj_stack = {
                "num_atoms": batch.num_atoms,
                "atom_types": batch.atom_types,
                "all_frac_coords": torch.stack(
                    [traj[i]["frac_coords"] for i in range(time_start, -1, -1)]
                ),
                "all_lattices": torch.stack(
                    [traj[i]["lattices"] for i in range(time_start, -1, -1)]
                ),
            }
        else:
            traj_stack = None

        return traj[0], traj_stack

    def on_after_backward(self):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here

        squared_norm = 0.0
        for parameter in self.decoder.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.data.norm(2).item() ** 2
        total_norm = squared_norm**0.5

        self.log_dict(
            {"grad_norm": total_norm},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch, batch_idx)

        loss_lattice = output_dict["loss_lattice"]
        loss_coord = output_dict["loss_coord"]
        loss = output_dict["loss"]

        self.log_dict(
            {
                "train_loss": loss,
                "lattice_loss": loss_lattice,
                "coord_loss": loss_coord,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan() or loss.isinf():
            print(batch_idx)
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix="val")

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix="test")

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):
        loss_lattice = output_dict["loss_lattice"]
        loss_coord = output_dict["loss_coord"]
        loss = output_dict["loss"]

        log_dict = {
            f"{prefix}_loss": loss,
            f"{prefix}_lattice_loss": loss_lattice,
            f"{prefix}_coord_loss": loss_coord,
        }

        return log_dict, loss
