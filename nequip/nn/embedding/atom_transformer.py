# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Any, Dict, List, Optional

import torch

from e3nn.o3._irreps import Irreps

from nequip.data import AtomicDataDict
from nequip.data.misc import chemical_symbols_to_atomic_numbers_dict
from nequip.nn.utils import scatter
from .._graph_mixin import GraphModuleMixin


class GaussianRBF(torch.nn.Module):
    """Gaussian radial basis encoding for scalar periodic-table features."""

    def __init__(
        self,
        num_basis: int,
        start: float,
        stop: float,
        gamma: Optional[float] = None,
    ):
        super().__init__()
        centers = torch.linspace(start, stop, num_basis)
        self.register_buffer("centers", centers)
        if gamma is None:
            spacing = (stop - start) / max(num_basis - 1, 1)
            gamma = 1.0 / max(spacing, 1e-6) ** 2
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)


class _MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AtomTransformerEmbed(GraphModuleMixin, torch.nn.Module):
    """Rich atom encoder followed by graph-wise Transformer mixing.

    This module replaces :class:`NodeTypeEmbed` when enabled. It first encodes
    each element from atomic number, periodic-table group/period, and atomic
    mass, then aggregates one-hop neighbor features from ``edge_index`` and
    fuses both branches before applying Transformer encoder layers within each
    graph in the batch. The output is written to ``node_attrs`` and
    ``node_features`` as scalar irreps, so the rest of NequIP can consume it
    unchanged.
    """

    def __init__(
        self,
        type_names: List[str],
        num_features: int,
        max_atomic_number: int = 118,
        z_embedding_dim: int = 32,
        rbf_dim: int = 16,
        atom_mlp_hidden_dim: Optional[int] = None,
        neighbor_mlp_hidden_dim: Optional[int] = None,
        fusion_mlp_hidden_dim: Optional[int] = None,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 4,
        transformer_ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        set_features: bool = True,
        irreps_in: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        assert num_features > 0, "`num_features` must be positive"
        assert max_atomic_number > 0, "`max_atomic_number` must be positive"
        assert transformer_num_layers >= 0, "`transformer_num_layers` must be >= 0"
        if transformer_num_layers > 0:
            assert num_features % transformer_num_heads == 0, (
                "`num_features` must be divisible by `transformer_num_heads`"
            )

        irreps_in = {} if irreps_in is None else dict(irreps_in)
        irreps_in[AtomicDataDict.ATOMIC_NUMBERS_KEY] = None

        self.num_types = len(type_names)
        self.num_features = num_features
        self.max_atomic_number = max_atomic_number
        self.set_features = set_features

        type_to_atomic_number = []
        for type_name in type_names:
            type_to_atomic_number.append(
                chemical_symbols_to_atomic_numbers_dict.get(type_name, 0)
            )
        self.register_buffer(
            "type_to_atomic_number",
            torch.as_tensor(type_to_atomic_number, dtype=torch.long),
        )

        self.z_embedding = torch.nn.Embedding(
            max_atomic_number + 1,
            z_embedding_dim,
            padding_idx=0,
        )
        self.group_rbf = GaussianRBF(rbf_dim, start=1.0, stop=18.0)
        self.period_rbf = GaussianRBF(rbf_dim, start=1.0, stop=7.0)

        group, period = self._build_periodic_table(max_atomic_number)
        atomic_mass = self._build_atomic_mass_table(max_atomic_number)
        self.register_buffer("group_table", group)
        self.register_buffer("period_table", period)
        self.register_buffer("atomic_mass_table", atomic_mass)
        self.register_buffer("mass_scale", torch.log1p(atomic_mass.max().clamp_min(1.0)))

        atom_mlp_hidden_dim = atom_mlp_hidden_dim or num_features * 2
        atom_input_dim = z_embedding_dim + 2 * rbf_dim + 1
        self.atom_mlp = _MLP(
            input_dim=atom_input_dim,
            hidden_dim=atom_mlp_hidden_dim,
            output_dim=num_features,
            dropout=dropout,
        )

        neighbor_mlp_hidden_dim = neighbor_mlp_hidden_dim or num_features * 2
        self.neighbor_phi = _MLP(
            input_dim=num_features,
            hidden_dim=neighbor_mlp_hidden_dim,
            output_dim=num_features,
            dropout=dropout,
        )
        self.neighbor_mlp = _MLP(
            input_dim=3 * num_features + 1,
            hidden_dim=neighbor_mlp_hidden_dim,
            output_dim=num_features,
            dropout=dropout,
        )

        fusion_mlp_hidden_dim = fusion_mlp_hidden_dim or num_features * 2
        self.fusion_mlp = _MLP(
            input_dim=2 * num_features,
            hidden_dim=fusion_mlp_hidden_dim,
            output_dim=num_features,
            dropout=dropout,
        )

        transformer_ffn_dim = transformer_ffn_dim or num_features * 4
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = (
            torch.nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
            if transformer_num_layers > 0
            else None
        )

        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(num_features, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        atomic_numbers = self._get_atomic_numbers(data)
        atom_features = self._encode_atoms(atomic_numbers)
        neighbor_features = self._encode_neighbors(atom_features, data)
        embedding = self.fusion_mlp(torch.cat([atom_features, neighbor_features], dim=-1))
        embedding = self._apply_transformer(embedding, data)

        data[AtomicDataDict.NODE_ATTRS_KEY] = embedding
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = embedding
        return data

    def _get_atomic_numbers(self, data: AtomicDataDict.Type) -> torch.Tensor:
        if AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
            atomic_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY].view(-1)
        else:
            atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].view(-1)
            atomic_numbers = torch.index_select(self.type_to_atomic_number, 0, atom_types)
            if atomic_numbers.numel() > 0 and atomic_numbers.min() == 0:
                raise ValueError(
                    "`AtomTransformerEmbed` needs either "
                    f"`{AtomicDataDict.ATOMIC_NUMBERS_KEY}` in the input data or "
                    "chemical element symbols in `type_names`."
                )

        if atomic_numbers.dtype != torch.long:
            atomic_numbers = atomic_numbers.long()
        if atomic_numbers.numel() > 0:
            if atomic_numbers.min() < 0 or atomic_numbers.max() > self.max_atomic_number:
                raise ValueError(
                    f"`{AtomicDataDict.ATOMIC_NUMBERS_KEY}` must be in "
                    f"[0, {self.max_atomic_number}]"
                )
        return atomic_numbers

    def _encode_atoms(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        z_embedding = self.z_embedding(atomic_numbers)
        group = self.group_table[atomic_numbers].type_as(z_embedding)
        period = self.period_table[atomic_numbers].type_as(z_embedding)
        group_features = self.group_rbf(group)
        period_features = self.period_rbf(period)
        mass_features = (
            torch.log1p(self.atomic_mass_table[atomic_numbers].clamp_min(0.0))
            / self.mass_scale
        ).type_as(z_embedding)

        features = torch.cat(
            [
                z_embedding,
                group_features,
                period_features,
                mass_features.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = self.atom_mlp(features)
        return features.masked_fill((atomic_numbers == 0).unsqueeze(-1), 0.0)

    def _encode_neighbors(
        self,
        atom_features: torch.Tensor,
        data: AtomicDataDict.Type,
    ) -> torch.Tensor:
        num_nodes = atom_features.size(0)
        feature_dim = atom_features.size(1)
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_dst = edge_index[0].view(-1)
        edge_src = edge_index[1].view(-1)

        if edge_src.numel() == 0:
            neighbor_inputs = atom_features.new_zeros((num_nodes, 3 * feature_dim + 1))
            return self.neighbor_mlp(neighbor_inputs)

        messages = self.neighbor_phi(torch.index_select(atom_features, 0, edge_src))
        sum_messages = scatter(messages, edge_dst, dim=0, dim_size=num_nodes)
        power_sum_messages = scatter(messages.square(), edge_dst, dim=0, dim_size=num_nodes)
        degree = scatter(
            torch.ones((edge_dst.size(0), 1), dtype=atom_features.dtype, device=atom_features.device),
            edge_dst,
            dim=0,
            dim_size=num_nodes,
        )
        mean_messages = sum_messages / degree.clamp_min(1.0)
        neighbor_inputs = torch.cat(
            [
                sum_messages,
                power_sum_messages,
                mean_messages,
                torch.log1p(degree),
            ],
            dim=-1,
        )
        return self.neighbor_mlp(neighbor_inputs)

    def _apply_transformer(
        self,
        embedding: torch.Tensor,
        data: AtomicDataDict.Type,
    ) -> torch.Tensor:
        if self.transformer is None or embedding.size(0) == 0:
            return embedding

        if AtomicDataDict.BATCH_KEY not in data:
            return self.transformer(embedding.unsqueeze(0)).squeeze(0)

        batch = data[AtomicDataDict.BATCH_KEY].view(-1)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        output = torch.empty_like(embedding)
        for graph_idx in range(num_graphs):
            mask = batch == graph_idx
            graph_embedding = embedding[mask]
            if graph_embedding.numel() == 0:
                continue
            output[mask] = self.transformer(graph_embedding.unsqueeze(0)).squeeze(0)
        return output

    def _build_atomic_mass_table(self, max_atomic_number: int) -> torch.Tensor:
        masses = [
            0.0,
            1.0080, 4.00260, 7.0, 9.012183, 10.81, 12.011, 14.007, 15.999, 18.99840316, 20.180,
            22.9897693, 24.305, 26.981538, 28.085, 30.97376200, 32.07, 35.45, 39.9, 39.0983, 40.08,
            44.95591, 47.867, 50.9415, 51.996, 54.93804, 55.84, 58.93319, 58.693, 63.55, 65.4,
            69.723, 72.63, 74.92159, 78.97, 79.90, 83.80, 85.468, 87.62, 88.90584, 91.22,
            92.90637, 95.95, 96.90636, 101.1, 102.9055, 106.42, 107.868, 112.41, 114.818, 118.71,
            121.760, 127.6, 126.9045, 131.29, 132.9054520, 137.33, 138.9055, 140.116, 140.90766, 144.24,
            144.91276, 150.4, 151.964, 157.25, 158.92535, 162.500, 164.93033, 167.26, 168.93422, 173.05,
            174.9667, 178.49, 180.9479, 183.84, 186.207, 190.2, 192.22, 195.08, 196.96657, 200.59,
            204.383, 207.0, 208.98040, 208.98243, 209.98715, 222.01758, 223.01973, 226.02541, 227.02775,
            232.038, 231.03588, 238.0289, 237.048172, 244.06420, 243.061380, 247.07035, 247.07031,
            251.07959, 252.0830, 257.09511, 258.09843, 259.10100, 266.120, 267.122, 268.126, 269.128,
            270.133, 269.1336, 277.154, 282.166, 282.169, 286.179, 286.182, 290.192, 290.196, 293.205,
            294.211, 295.216,
        ]
        table = torch.zeros(max_atomic_number + 1, dtype=torch.float)
        supported = min(max_atomic_number, len(masses) - 1)
        table[: supported + 1] = torch.tensor(masses[: supported + 1], dtype=torch.float)
        return table

    def _build_periodic_table(self, max_atomic_number: int):
        group = torch.zeros(max_atomic_number + 1, dtype=torch.float)
        period = torch.zeros(max_atomic_number + 1, dtype=torch.float)
        table = {
            1: (1, 1), 2: (18, 1),
            3: (1, 2), 4: (2, 2), 5: (13, 2), 6: (14, 2), 7: (15, 2), 8: (16, 2), 9: (17, 2), 10: (18, 2),
            11: (1, 3), 12: (2, 3), 13: (13, 3), 14: (14, 3), 15: (15, 3), 16: (16, 3), 17: (17, 3), 18: (18, 3),
            19: (1, 4), 20: (2, 4), 21: (3, 4), 22: (4, 4), 23: (5, 4), 24: (6, 4), 25: (7, 4), 26: (8, 4), 27: (9, 4), 28: (10, 4), 29: (11, 4), 30: (12, 4), 31: (13, 4), 32: (14, 4), 33: (15, 4), 34: (16, 4), 35: (17, 4), 36: (18, 4),
            37: (1, 5), 38: (2, 5), 39: (3, 5), 40: (4, 5), 41: (5, 5), 42: (6, 5), 43: (7, 5), 44: (8, 5), 45: (9, 5), 46: (10, 5), 47: (11, 5), 48: (12, 5), 49: (13, 5), 50: (14, 5), 51: (15, 5), 52: (16, 5), 53: (17, 5), 54: (18, 5),
            55: (1, 6), 56: (2, 6), 57: (3, 6), 58: (3, 6), 59: (3, 6), 60: (3, 6), 61: (3, 6), 62: (3, 6), 63: (3, 6), 64: (3, 6), 65: (3, 6), 66: (3, 6), 67: (3, 6), 68: (3, 6), 69: (3, 6), 70: (3, 6), 71: (3, 6), 72: (4, 6), 73: (5, 6), 74: (6, 6), 75: (7, 6), 76: (8, 6), 77: (9, 6), 78: (10, 6), 79: (11, 6), 80: (12, 6), 81: (13, 6), 82: (14, 6), 83: (15, 6), 84: (16, 6), 85: (17, 6), 86: (18, 6),
            87: (1, 7), 88: (2, 7), 89: (3, 7), 90: (3, 7), 91: (3, 7), 92: (3, 7), 93: (3, 7), 94: (3, 7), 95: (3, 7), 96: (3, 7), 97: (3, 7), 98: (3, 7), 99: (3, 7), 100: (3, 7), 101: (3, 7), 102: (3, 7), 103: (3, 7), 104: (4, 7), 105: (5, 7), 106: (6, 7), 107: (7, 7), 108: (8, 7), 109: (9, 7), 110: (10, 7), 111: (11, 7), 112: (12, 7), 113: (13, 7), 114: (14, 7), 115: (15, 7), 116: (16, 7), 117: (17, 7), 118: (18, 7),
        }
        for atomic_number, (atom_group, atom_period) in table.items():
            if atomic_number > max_atomic_number:
                break
            group[atomic_number] = float(atom_group)
            period[atomic_number] = float(atom_period)
        return group, period
