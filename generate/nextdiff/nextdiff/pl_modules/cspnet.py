"""Graph decoder used by the NextDiff sampling model."""

import math

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter


MAX_ATOMIC_NUM = 100


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(n_frequencies)
        self.dim = n_frequencies * 2 * n_space

    def forward(self, coordinates):
        phases = coordinates.unsqueeze(-1) * self.frequencies[None, None, :].to(
            coordinates.device
        )
        phases = phases.reshape(-1, self.n_frequencies * self.n_space)
        return torch.cat((phases.sin(), phases.cos()), dim=-1).detach()


class CSPLayer(nn.Module):
    """A single message-passing layer for :class:`CSPNet`."""

    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ln=False,
        ip=True,
    ):
        super().__init__()
        self.dis_dim = dis_emb.dim if dis_emb is not None else 3
        self.dis_emb = dis_emb
        self.ip = True
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattice_rep,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        source, target = edge_index
        source_features = node_features[source]
        target_features = node_features[target]

        if frac_diff is None:
            frac_diff = (frac_coords[target] - frac_coords[source]) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)

        edge_inputs = torch.cat(
            (
                source_features,
                target_features,
                lattice_rep[edge2graph],
                frac_diff,
            ),
            dim=1,
        )
        return self.edge_mlp(edge_inputs)

    def node_model(self, node_features, edge_features, edge_index):
        aggregated = scatter(
            edge_features,
            edge_index[0],
            dim=0,
            reduce="mean",
            dim_size=node_features.shape[0],
        )
        return self.node_mlp(torch.cat((node_features, aggregated), dim=1))

    def forward(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        residual = node_features
        normalized = self.layer_norm(residual) if self.ln else residual
        edge_features = self.edge_model(
            normalized,
            frac_coords,
            lattices,
            edge_index,
            edge2graph,
            frac_diff,
        )
        return residual + self.node_model(normalized, edge_features, edge_index)


class CSPNet(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        num_layers=4,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        coord_style="node",
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        attn=False,
        pred_type=False,
        coord_cart=False,
        dense=False,
        smooth=False,
        ip=True,
        gate=False,
        pred_scalar=False,
        pooling="mean",
        cross_attention=False,
        cross_attention_heads=4,
        cross_attention_dropout=0.0,
    ):
        super().__init__()

        self.smooth = smooth
        self.ip = ip
        self.node_embedding = (
            nn.Linear(max_atoms, hidden_dim)
            if smooth
            else nn.Embedding(max_atoms, hidden_dim)
        )

        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None

        self.coord_style = coord_style
        for layer_index in range(num_layers):
            self.add_module(
                f"csp_layer_{layer_index}",
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip),
            )
        self.num_layers = num_layers
        self.dense = dense

        hidden_dim_before_out = hidden_dim * (num_layers + 1) if dense else hidden_dim
        self.coord_out = nn.Linear(hidden_dim_before_out, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim_before_out, 6, bias=False)

        self.edge_style = edge_style
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        if ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.pred_type = pred_type
        if pred_type:
            self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

        self.pred_scalar = pred_scalar
        if pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim_before_out, 1)
        self.pooling = pooling

        # The optional path is deliberately disabled by default. Consequently,
        # legacy checkpoints instantiate no additional parameters and retain an
        # identical state_dict. When enabled for a newly trained model, final
        # node states query the initial atom/time embeddings of the same graph.
        self.cross_attention_gate = bool(cross_attention)
        self.cross_attention = None
        self.cross_attention_norm = None
        if self.cross_attention_gate:
            if hidden_dim % cross_attention_heads:
                raise ValueError(
                    "hidden_dim must be divisible by cross_attention_heads"
                )
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim,
                cross_attention_heads,
                dropout=cross_attention_dropout,
                batch_first=True,
            )
            self.cross_attention_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def gen_edges(num_atoms, frac_coords):
        graph_blocks = [
            torch.ones(atom_count, atom_count, device=num_atoms.device)
            for atom_count in num_atoms
        ]
        edge_index, _ = dense_to_sparse(torch.block_diag(*graph_blocks))
        frac_diff = (frac_coords[edge_index[1]] - frac_coords[edge_index[0]]) % 1.0
        return edge_index, frac_diff

    def _apply_cross_attention(self, query_features, context_features, num_atoms):
        if not self.cross_attention_gate:
            return query_features

        atom_counts = tuple(int(count) for count in num_atoms.tolist())
        query_groups = torch.split(query_features, atom_counts)
        context_groups = torch.split(context_features, atom_counts)
        attended_groups = []

        for queries, context in zip(query_groups, context_groups):
            attended, _ = self.cross_attention(
                queries.unsqueeze(0),
                context.unsqueeze(0),
                context.unsqueeze(0),
                need_weights=False,
            )
            attended_groups.append(
                self.cross_attention_norm(queries + attended.squeeze(0))
            )
        return torch.cat(attended_groups, dim=0)

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]

        node_features = (
            self.node_embedding(atom_types)
            if self.smooth
            else self.node_embedding(atom_types - 1)
        )
        time_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = self.atom_latent_emb(
            torch.cat((node_features, time_per_atom), dim=1)
        )

        context_features = node_features
        hidden_states = [context_features]
        for layer_index in range(self.num_layers):
            layer = getattr(self, f"csp_layer_{layer_index}")
            node_features = layer(
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )
            if layer_index != self.num_layers - 1:
                hidden_states.append(node_features)

        if self.ln:
            node_features = self.final_layer_norm(node_features)
        node_features = self._apply_cross_attention(
            node_features, context_features, num_atoms
        )
        hidden_states.append(node_features)

        if self.dense:
            node_features = torch.cat(hidden_states, dim=-1)
        graph_features = scatter(
            node_features,
            node2graph,
            dim=0,
            reduce=self.pooling,
        )

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        coord_out = self.coord_out(node_features)
        lattice_out = self.lattice_out(graph_features)
        if self.pred_type:
            return lattice_out, coord_out, self.type_out(node_features)
        return lattice_out, coord_out
