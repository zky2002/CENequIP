import torch

from nequip.data import AtomicDataDict
from nequip.model.nequip_models import NequIPGNNModel
from nequip.nn.embedding import AtomTransformerEmbed
from nequip.utils.global_state import set_global_state


def test_atom_transformer_embed_forward_with_atomic_numbers():
    module = AtomTransformerEmbed(
        type_names=["H", "C", "O"],
        num_features=8,
        z_embedding_dim=4,
        rbf_dim=3,
        transformer_num_layers=1,
        transformer_num_heads=2,
    )
    module.eval()

    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0], [1], [2], [0]]),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.tensor([[1], [6], [8], [1]]),
        AtomicDataDict.POSITIONS_KEY: torch.randn(4, 3),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1, 2, 1], [1, 0, 1, 2]],
            dtype=torch.long,
        ),
        AtomicDataDict.BATCH_KEY: torch.tensor([0, 0, 0, 1], dtype=torch.long),
    }

    out = module(data)

    assert out[AtomicDataDict.NODE_ATTRS_KEY].shape == (4, 8)
    assert out[AtomicDataDict.NODE_FEATURES_KEY].shape == (4, 8)
    assert torch.isfinite(out[AtomicDataDict.NODE_FEATURES_KEY]).all()


def test_atom_transformer_embed_falls_back_to_type_names():
    module = AtomTransformerEmbed(
        type_names=["H", "C"],
        num_features=4,
        z_embedding_dim=4,
        rbf_dim=2,
        transformer_num_layers=0,
        transformer_num_heads=1,
    )

    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0], [1]]),
        AtomicDataDict.POSITIONS_KEY: torch.randn(2, 3),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0], [1]], dtype=torch.long),
    }

    out = module(data)

    assert out[AtomicDataDict.NODE_FEATURES_KEY].shape == (2, 4)


def test_nequip_model_accepts_atom_transformer_embed():
    set_global_state(allow_tf32=False)
    model = NequIPGNNModel(
        seed=123,
        model_dtype="float32",
        type_names=["H", "C", "O"],
        r_max=4.0,
        avg_num_neighbors=2.0,
        num_layers=1,
        l_max=1,
        parity=False,
        num_features=4,
        type_embed_num_features=8,
        radial_mlp_depth=1,
        radial_mlp_width=8,
        per_type_energy_shifts={"H": 0.0, "C": 0.0, "O": 0.0},
        do_derivatives=False,
        atom_transformer_embed=True,
        atom_transformer_kwargs={
            "z_embedding_dim": 4,
            "rbf_dim": 2,
            "transformer_num_layers": 1,
            "transformer_num_heads": 2,
        },
    )
    model.eval()

    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0], [1], [2]]),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.tensor([[1], [6], [8]]),
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=torch.long,
        ),
    }

    out = model(data)

    assert AtomicDataDict.PER_ATOM_ENERGY_KEY in out
    assert AtomicDataDict.TOTAL_ENERGY_KEY in out
    assert out[AtomicDataDict.PER_ATOM_ENERGY_KEY].shape == (3, 1)


if __name__ == "__main__":
    test_atom_transformer_embed_forward_with_atomic_numbers()
    test_atom_transformer_embed_falls_back_to_type_names()
    test_nequip_model_accepts_atom_transformer_embed()
