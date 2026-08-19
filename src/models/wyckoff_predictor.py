# path: src/models/wyckoff_predictor.py
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.chemistry import ELEMENT_ORDER_DICT


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * d_model),
        )

    def forward(self, x, cond_vec):
        gamma, beta = self.net(cond_vec).chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class ConcatConditioner(nn.Module):
    """Naive conditioning baseline: concatenate SG to every token and project."""

    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim + d_model, d_model)

    def forward(self, x, cond_vec):
        cond = cond_vec.unsqueeze(1).expand(-1, x.size(1), -1)
        return self.proj(torch.cat([x, cond], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(d_model, dim_ff)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.lin2(x)
        return x


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryPositionalEmbeddingHead(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv)

    def get_cos_sin(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


class RopeElementGatedMHA(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, vocab_size: int, pad_idx: int = 0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.gate_q = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.gate_kv = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        nn.init.zeros_(self.gate_q.weight)
        nn.init.zeros_(self.gate_kv.weight)

    def _prep_rope_cos_sin(self, cos, sin, L: int, dtype: torch.dtype, device: torch.device):
        def _to_head_shape(t):
            t = t.to(dtype=dtype, device=device)
            if t.dim() == 3:
                _, Lc, Dc = t.shape
                H = self.nhead
                d = self.head_dim
                if Dc != H * d:
                    raise RuntimeError(f"RoPE cos/sin last dim {Dc} != {H*d}")
                t = t.view(1, Lc, H, d).permute(0, 2, 1, 3).contiguous()
                return t
            elif t.dim() == 4:
                if t.size(1) == 1:
                    t = t.expand(1, self.nhead, L, self.head_dim)
                else:
                    assert (
                        t.size(1) == self.nhead and t.size(2) == L and t.size(3) == self.head_dim
                    ), f"Expected [1,H,L,d], got {tuple(t.shape)}"
                return t
            else:
                raise RuntimeError(f"Unsupported RoPE dim: {t.dim()}")

        return _to_head_shape(cos), _to_head_shape(sin)

    def forward(self, x, token_ids, cos, sin, key_padding_mask=None):
        B, L, D = x.shape

        gq = torch.tanh(self.gate_q(token_ids))
        gkv = torch.tanh(self.gate_kv(token_ids))
        xq = x * (1.0 + gq)
        xkv = x * (1.0 + gkv)

        q = self.q_proj(xq)
        k = self.k_proj(xkv)
        v = self.v_proj(xkv)

        def split_heads(t):
            return t.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        cos_h, sin_h = self._prep_rope_cos_sin(cos, sin, L=L, dtype=q.dtype, device=q.device)
        q = (q * cos_h) + (_rotate_half(q) * sin_h)
        k = (k * cos_h) + (_rotate_half(k) * sin_h)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        out = self.proj_drop(out)
        return out, None


class SwitchFFNExpert(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x):
        return self.ffn(x)


class ImprovedSwitchMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_ff: int,
        num_experts: int = 8,
        dropout: float = 0.2,
        capacity_factor: float = 1.25,
        router_noisy_std: float = 0.0,
        top_k: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_ff = dim_ff
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.top_k = top_k
        self.dropout = dropout
        self.use_grouped_gemm = True

        self.router = nn.Linear(d_model, num_experts, bias=False)
        # Store expert matrices in a batched layout.  This preserves sparse
        # top-k routing while executing the selected expert MLPs with two
        # grouped batched GEMMs rather than 2*num_experts small kernel launches.
        self.expert_weight1 = nn.Parameter(torch.empty(num_experts, dim_ff, d_model))
        self.expert_bias1 = nn.Parameter(torch.empty(num_experts, dim_ff))
        self.expert_weight2 = nn.Parameter(torch.empty(num_experts, d_model, dim_ff))
        self.expert_bias2 = nn.Parameter(torch.empty(num_experts, d_model))
        self.router_noisy_std = router_noisy_std
        self.last_requested_assignments = 0
        self.last_processed_assignments = 0
        self.last_batched_slots = 0
        self._reset_expert_parameters()
        nn.init.xavier_uniform_(self.router.weight)

    def _reset_expert_parameters(self):
        for expert_index in range(self.num_experts):
            nn.init.kaiming_uniform_(
                self.expert_weight1[expert_index], a=math.sqrt(5)
            )
            bound1 = 1 / math.sqrt(self.d_model)
            nn.init.uniform_(self.expert_bias1[expert_index], -bound1, bound1)
            nn.init.kaiming_uniform_(
                self.expert_weight2[expert_index], a=math.sqrt(5)
            )
            bound2 = 1 / math.sqrt(self.dim_ff)
            nn.init.uniform_(self.expert_bias2[expert_index], -bound2, bound2)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load both the grouped layout and legacy ModuleList checkpoints."""
        grouped_key = prefix + "expert_weight1"
        legacy_keys = []
        if grouped_key not in state_dict:
            key_sets = {
                "expert_weight1": [
                    prefix + f"experts.{e}.ffn.0.weight"
                    for e in range(self.num_experts)
                ],
                "expert_bias1": [
                    prefix + f"experts.{e}.ffn.0.bias"
                    for e in range(self.num_experts)
                ],
                "expert_weight2": [
                    prefix + f"experts.{e}.ffn.3.weight"
                    for e in range(self.num_experts)
                ],
                "expert_bias2": [
                    prefix + f"experts.{e}.ffn.3.bias"
                    for e in range(self.num_experts)
                ],
            }
            if all(key in state_dict for keys in key_sets.values() for key in keys):
                for new_name, keys in key_sets.items():
                    state_dict[prefix + new_name] = torch.stack(
                        [state_dict[key] for key in keys], dim=0
                    )
                    legacy_keys.extend(keys)
        for key in legacy_keys:
            state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def _compute_capacity(self, n_tokens: int):
        base = math.ceil(n_tokens / max(1, self.num_experts))
        return max(1, int(self.capacity_factor * base))

    def _selected_assignments(self, expert_idxs, gate_vals, capacity):
        selected = []
        for expert_index in range(self.num_experts):
            token_parts, gate_parts = [], []
            if self.top_k > 1:
                for route_index in range(self.top_k):
                    token_indices = (
                        expert_idxs[:, route_index] == expert_index
                    ).nonzero(as_tuple=False).squeeze(-1)
                    if token_indices.numel() > 0:
                        token_parts.append(token_indices)
                        gate_parts.append(
                            gate_vals[:, route_index]
                            .index_select(0, token_indices)
                            .unsqueeze(-1)
                        )
            else:
                token_indices = (expert_idxs == expert_index).nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if token_indices.numel() > 0:
                    token_parts.append(token_indices)
                    gate_parts.append(
                        gate_vals.index_select(0, token_indices).unsqueeze(-1)
                    )

            if not token_parts:
                selected.append((None, None))
                continue
            token_indices = torch.cat(token_parts)
            gates = torch.cat(gate_parts)
            if token_indices.numel() > capacity:
                keep = torch.topk(gates.squeeze(-1), k=capacity, dim=0).indices
                token_indices = token_indices.index_select(0, keep)
                gates = gates.index_select(0, keep)
            selected.append((token_indices, gates))
        return selected

    def _reference_expert_forward(self, x_valid, selected):
        y_valid = torch.zeros_like(x_valid)
        processed = 0
        for expert_index, (token_indices, gates) in enumerate(selected):
            if token_indices is None:
                continue
            expert_input = x_valid.index_select(0, token_indices)
            hidden = F.linear(
                expert_input,
                self.expert_weight1[expert_index],
                self.expert_bias1[expert_index],
            )
            hidden = F.gelu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            expert_output = F.linear(
                hidden,
                self.expert_weight2[expert_index],
                self.expert_bias2[expert_index],
            )
            y_valid.index_add_(0, token_indices, expert_output * gates)
            processed += int(token_indices.numel())
        self.last_processed_assignments = processed
        self.last_batched_slots = processed
        return y_valid

    def _grouped_expert_forward(
        self,
        x_valid,
        expert_idxs,
        gate_vals,
        capacity,
    ):
        # Sort assignments by expert and descending gate in two stable passes.
        # This implements the same per-expert capacity rule as the reference
        # loop without launching nonzero/index kernels separately for every
        # expert and every route.
        token_count = x_valid.size(0)
        flat_experts = expert_idxs.reshape(-1)
        flat_gates = gate_vals.reshape(-1)
        flat_tokens = (
            torch.arange(token_count, device=x_valid.device)
            .unsqueeze(1)
            .expand(token_count, self.top_k)
            .reshape(-1)
        )
        by_gate = torch.argsort(flat_gates, descending=True, stable=True)
        by_expert = torch.argsort(flat_experts[by_gate], stable=True)
        order = by_gate[by_expert]
        sorted_experts = flat_experts[order]
        sorted_gates = flat_gates[order]
        sorted_tokens = flat_tokens[order]

        counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        starts = counts.cumsum(0) - counts
        positions = torch.arange(
            sorted_experts.numel(), device=x_valid.device
        ) - torch.repeat_interleave(
            starts, counts, output_size=sorted_experts.numel()
        )
        keep = positions < capacity
        selected_experts = sorted_experts[keep]
        selected_positions = positions[keep]
        selected_gates = sorted_gates[keep]
        selected_tokens = sorted_tokens[keep]
        # Use the precomputed capacity as the grouped-GEMM width.  Besides
        # making allocation deterministic, this avoids a device-to-host
        # synchronization on counts.max().  Unoccupied slots are masked before
        # the final scatter and therefore do not affect outputs.
        max_slots = capacity
        if max_slots == 0:
            self.last_processed_assignments = 0
            self.last_batched_slots = 0
            return torch.zeros_like(x_valid)

        expert_inputs = x_valid.new_zeros(
            (self.num_experts, max_slots, self.d_model)
        )
        slot_gates = x_valid.new_zeros((self.num_experts, max_slots, 1))
        slot_tokens = torch.full(
            (self.num_experts, max_slots),
            -1,
            dtype=torch.long,
            device=x_valid.device,
        )
        expert_inputs[selected_experts, selected_positions] = x_valid.index_select(
            0, selected_tokens
        )
        slot_gates[selected_experts, selected_positions, 0] = selected_gates
        slot_tokens[selected_experts, selected_positions] = selected_tokens

        hidden = torch.bmm(
            expert_inputs, self.expert_weight1.transpose(1, 2)
        ) + self.expert_bias1.unsqueeze(1)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        expert_outputs = torch.bmm(
            hidden, self.expert_weight2.transpose(1, 2)
        ) + self.expert_bias2.unsqueeze(1)
        expert_outputs = expert_outputs * slot_gates

        occupied = slot_tokens.ge(0)
        y_valid = torch.zeros_like(x_valid)
        y_valid.index_add_(
            0,
            slot_tokens[occupied],
            expert_outputs[occupied],
        )
        self.last_processed_assignments = int(selected_tokens.numel())
        self.last_batched_slots = self.num_experts * max_slots
        return y_valid

    def forward(self, x, valid_mask=None):
        B, L, D = x.shape
        device = x.device

        if valid_mask is None:
            valid_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        flat_x = x.reshape(B * L, D)
        flat_mask = valid_mask.reshape(B * L)
        idx_valid = flat_mask.nonzero(as_tuple=False).squeeze(-1)

        if idx_valid.numel() == 0:
            self.last_requested_assignments = 0
            self.last_processed_assignments = 0
            self.last_batched_slots = 0
            return torch.zeros_like(x), x.new_tensor(0.0)

        x_valid = flat_x.index_select(0, idx_valid)
        logits = self.router(x_valid)
        # Router noise is a training regularizer.  Applying it during
        # ``model.eval()`` makes repeated inference runs non-deterministic and
        # changes the beam-search objective from run to run.
        if self.training and self.router_noisy_std > 0:
            logits = logits + self.router_noisy_std * torch.randn_like(logits)
        probs = F.softmax(logits, dim=-1)

        gate_vals, expert_idxs = torch.topk(probs, k=self.top_k, dim=-1)
        if self.top_k > 1:
            gate_vals = gate_vals / gate_vals.sum(dim=-1, keepdim=True)

        N = x_valid.size(0)
        E = self.num_experts

        mean_prob = probs.mean(dim=0)
        with torch.no_grad():
            counts = torch.bincount(
                expert_idxs.reshape(-1), minlength=E
            ).float()
        frac_tok = counts / max(1.0, float(N * self.top_k))
        aux_loss = (E * (mean_prob * frac_tok).sum()).to(x.dtype)

        capacity = self._compute_capacity(N * self.top_k)
        self.last_requested_assignments = int(N * self.top_k)
        if self.use_grouped_gemm:
            y_valid = self._grouped_expert_forward(
                x_valid,
                expert_idxs,
                gate_vals,
                capacity,
            )
        else:
            reference_expert_idxs = (
                expert_idxs.squeeze(-1) if self.top_k == 1 else expert_idxs
            )
            reference_gate_vals = (
                gate_vals.squeeze(-1) if self.top_k == 1 else gate_vals
            )
            selected = self._selected_assignments(
                reference_expert_idxs, reference_gate_vals, capacity
            )
            y_valid = self._reference_expert_forward(x_valid, selected)

        flat_out = torch.zeros_like(flat_x)
        flat_out.index_copy_(0, idx_valid, y_valid)
        return flat_out.view(B, L, D), aux_loss


def z_to_period(z_long: torch.Tensor):
    bins = torch.tensor([0, 2, 10, 18, 36, 54, 86, 118], dtype=torch.long, device=z_long.device)
    p = torch.bucketize(z_long.clamp(min=0), bins)
    p = torch.where(z_long > 0, p, torch.zeros_like(p))
    return p


class AtomicFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.atomic_num_emb = nn.Embedding(119, d_model // 3)
        self.group_emb = nn.Embedding(19, d_model // 3)
        self.period_emb = nn.Embedding(8, d_model // 3)
        self.fc = nn.Linear((d_model // 3) * 3, d_model)

    def forward(self, atomic_nums, groups, periods):
        e1 = self.atomic_num_emb(atomic_nums)
        e2 = self.group_emb(groups)
        e3 = self.period_emb(periods)
        return self.fc(torch.cat([e1, e2, e3], dim=-1))


class AtomicLookup(nn.Module):
    def __init__(self, vocab_size: int, formula_vocab: dict | None, element_order_dict: dict):
        super().__init__()
        z_table = torch.zeros(vocab_size, dtype=torch.long)
        if formula_vocab is not None:
            inv = {v: k for k, v in formula_vocab.items()}
            for idx in range(vocab_size):
                sym = inv.get(idx, "X")
                z_table[idx] = int(element_order_dict.get(sym, 0))
        period_table = z_to_period(z_table)
        group_table = torch.zeros_like(z_table)
        self.register_buffer("z_table", z_table, persistent=False)
        self.register_buffer("period_table", period_table, persistent=False)
        self.register_buffer("group_table", group_table, persistent=False)

    def forward(self, src):
        z = self.z_table[src]
        g = self.group_table[src]
        p = self.period_table[src]
        return z, g, p


class WyckoffPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        space_group_size: int,
        wyckoff_size: int,
        multiplicity_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        max_len: int = 512,
        dropout: float = 0.2,
        max_multiplicities: int = 30,
        use_moe: bool = True,
        moe_layers: str | list[int] = "all",
        num_experts: int = 8,
        capacity_factor: float = 1.25,
        router_noisy_std: float = 0.0,
        moe_loss_coef: float = 0.01,
        moe_top_k: int = 1,
        formula_vocab: Optional[dict[str, int]] = None,
        enable_sg_head: bool = True,
        conditioning_mode: str = "film",
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.max_multiplicities = max_multiplicities
        self.wyckoff_size = wyckoff_size
        self.use_moe = use_moe
        self.moe_loss_coef = moe_loss_coef
        self.enable_sg_head = enable_sg_head
        if conditioning_mode not in {"film", "concat"}:
            raise ValueError(f"Unsupported conditioning_mode={conditioning_mode!r}")
        self.conditioning_mode = conditioning_mode

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.space_group_embedding = nn.Embedding(space_group_size, d_model)
        self.multiplicity_embedding = nn.Embedding(multiplicity_vocab_size + 1, d_model, padding_idx=0)
        self.count_embedding = nn.Embedding(max_len + 1, d_model)

        self.rope = RotaryPositionalEmbeddingHead(head_dim=d_model // nhead, base=10000)

        self.atomic_lookup = AtomicLookup(vocab_size, formula_vocab, ELEMENT_ORDER_DICT)
        self.atomic_feature_emb = AtomicFeatureEmbedding(d_model)

        self.multiplicity_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mul_norm_in = RMSNorm(d_model)
        self.mul_norm_out = RMSNorm(d_model)

        self.mhas = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.film_attn = nn.ModuleList()
        self.film_ffn = nn.ModuleList()
        self.ls_attn = nn.ParameterList()
        self.ls_ffn = nn.ParameterList()

        def use_moe_on_layer(i):
            if not use_moe:
                return False
            if isinstance(moe_layers, (list, tuple, set)):
                return i in set(moe_layers)
            if moe_layers == "all":
                return True
            if moe_layers == "even":
                return i % 2 == 0
            if moe_layers == "odd":
                return i % 2 == 1
            if moe_layers == "last":
                return i == num_layers - 1
            return True

        for i in range(num_layers):
            self.mhas.append(RopeElementGatedMHA(d_model, nhead, dropout, vocab_size=vocab_size, pad_idx=0))
            if use_moe_on_layer(i):
                self.ffns.append(
                    ImprovedSwitchMoE(
                        d_model=d_model,
                        dim_ff=dim_feedforward,
                        num_experts=num_experts,
                        dropout=dropout,
                        capacity_factor=capacity_factor,
                        router_noisy_std=router_noisy_std,
                        top_k=moe_top_k,
                    )
                )
            else:
                self.ffns.append(FeedForward(d_model, dim_feedforward, dropout))

            self.norm1.append(RMSNorm(d_model))
            self.norm2.append(RMSNorm(d_model))
            conditioner = FiLM if conditioning_mode == "film" else ConcatConditioner
            if conditioning_mode == "film":
                self.film_attn.append(conditioner(d_model, d_model, hidden=max(128, d_model // 2)))
                self.film_ffn.append(conditioner(d_model, d_model, hidden=max(128, d_model // 2)))
            else:
                self.film_attn.append(conditioner(d_model, d_model))
                self.film_ffn.append(conditioner(d_model, d_model))
            self.ls_attn.append(nn.Parameter(1e-2 * torch.ones(1)))
            self.ls_ffn.append(nn.Parameter(1e-2 * torch.ones(1)))

        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(d_model, wyckoff_size)
        self.mul_prior = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, wyckoff_size),
        )

        if self.enable_sg_head:
            self.sg_classifier = nn.Linear(d_model, space_group_size)

        self._reset_parameters()
        self.moe_aux_loss = torch.tensor(0.0)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.space_group_embedding.weight)
        nn.init.xavier_uniform_(self.multiplicity_embedding.weight)
        nn.init.xavier_uniform_(self.count_embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        if self.enable_sg_head:
            nn.init.xavier_uniform_(self.sg_classifier.weight)
            nn.init.zeros_(self.sg_classifier.bias)

    def _encode_tokens(self, src, src_mask, space_group=None, multiplicities=None, multiplicity_mask=None):
        B, L = src.shape
        device = src.device

        x = self.embedding(src) * math.sqrt(self.d_model)
        z, g, p = self.atomic_lookup(src)
        feat = self.atomic_feature_emb(z, g, p)
        x = x + feat

        if space_group is not None:
            sg_cond = self.space_group_embedding(space_group)
        else:
            sg_cond = torch.zeros(B, self.d_model, device=device)

        n_valid = src_mask.long().sum(dim=1).clamp_(0, self.max_len)
        x = x + self.count_embedding(n_valid).unsqueeze(1)

        cos, sin = self.rope.get_cos_sin(L, device=device, dtype=x.dtype)

        if multiplicities is not None and multiplicity_mask is not None:
            x_in = self.mul_norm_in(x)
            mul_emb = self.multiplicity_embedding(multiplicities)
            mul_out, _ = self.multiplicity_attn(x_in, mul_emb, mul_emb, key_padding_mask=~multiplicity_mask)
            x = self.mul_norm_out(x + self.dropout(mul_out))

        key_padding_mask = ~src_mask
        total_aux = x.new_tensor(0.0)

        for i in range(len(self.mhas)):
            y = self.norm1[i](x)
            attn_out, _ = self.mhas[i](y, token_ids=src, cos=cos, sin=sin, key_padding_mask=key_padding_mask)
            attn_out = self.film_attn[i](attn_out, sg_cond)
            x = x + self.ls_attn[i] * self.dropout(attn_out)

            y = self.norm2[i](x)
            ffn = self.ffns[i]
            if isinstance(ffn, ImprovedSwitchMoE):
                ffn_out, aux = ffn(y, valid_mask=src_mask)
                total_aux = total_aux + aux
            else:
                ffn_out = ffn(y)
            ffn_out = self.film_ffn[i](ffn_out, sg_cond)
            x = x + self.ls_ffn[i] * self.dropout(ffn_out)

        self.moe_aux_loss = total_aux
        return x, total_aux

    def forward(self, src, src_mask, space_group, multiplicities, multiplicity_mask):
        x, total_aux = self._encode_tokens(src, src_mask, space_group, multiplicities, multiplicity_mask)
        x = self.dropout(x)
        logits = self.fc_out(x)

        mul_valid = multiplicity_mask.float().unsqueeze(-1)
        mul_feat = self.multiplicity_embedding(multiplicities) * mul_valid
        denom = mul_valid.sum(dim=1).clamp(min=1.0)
        mul_pool = mul_feat.sum(dim=1) / denom
        class_bias = self.mul_prior(mul_pool).unsqueeze(1)
        logits = logits + class_bias

        return logits, total_aux

    def forward_sg(self, src, src_mask):
        x, total_aux = self._encode_tokens(src, src_mask, None, None, None)
        denom = src_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        h_graph = (x * src_mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.sg_classifier(h_graph) if self.enable_sg_head else None
        return logits, total_aux

    @torch.no_grad()
    def extract_feature(self, src, src_mask, space_group, multiplicities, multiplicity_mask, pool: str = "mean"):
        x, _ = self._encode_tokens(src, src_mask, space_group, multiplicities, multiplicity_mask)
        if pool == "max":
            h_graph, _ = (x + (~src_mask).unsqueeze(-1) * (-1e9)).max(dim=1)
        else:
            denom = src_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            h_graph = (x * src_mask.unsqueeze(-1)).sum(dim=1) / denom
        return h_graph
