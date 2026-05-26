# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""Convert the PyG QM9 processed tensor file to an ASE extxyz file for NequIP.

The QM9 file commonly named ``qm9_v3.pt`` is the processed PyG InMemoryDataset
payload. NequIP's stock training script does not include a QM9 datamodule, so
this script turns one scalar QM9 target into ASE ``energy`` entries and writes
the molecular geometries as ``extxyz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


QM9_TARGETS: Dict[str, int] = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "zpve": 6,
    "u0": 7,
    "u": 8,
    "h": 9,
    "g": 10,
    "cv": 11,
    "u0_atom": 12,
    "u_atom": 13,
    "h_atom": 14,
    "g_atom": 15,
    "a": 16,
    "b": 17,
    "c": 18,
}


def _add_extra_site_packages(paths: Iterable[str]) -> None:
    for path in paths:
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _load_qm9(path: Path, extra_site_packages: Iterable[str]):
    _add_extra_site_packages(extra_site_packages)
    return torch.load(path, map_location="cpu", weights_only=False)


def _normalise_payload(payload):
    if isinstance(payload, tuple):
        if len(payload) >= 2:
            return payload[0], payload[1]
    if isinstance(payload, dict):
        if "data" in payload and "slices" in payload:
            return payload["data"], payload["slices"]
        print(f"QM9 payload dict keys: {list(payload.keys())}", flush=True)
    elif isinstance(payload, list):
        print(f"QM9 payload list length: {len(payload)}", flush=True)
        print(
            "QM9 payload list element types: "
            f"{[type(item).__name__ for item in payload[:8]]}",
            flush=True,
        )
        if len(payload) > 0 and isinstance(payload[0], dict):
            print(f"QM9 first sample keys: {list(payload[0].keys())}", flush=True)
    else:
        print(f"QM9 payload type: {type(payload)!r}", flush=True)
    raise TypeError(
        "Unsupported QM9 payload. Expected a PyG InMemoryDataset tuple "
        "or a dict containing `data` and `slices`."
    )


def _slice_item(data, slices, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = data.z[slices["z"][index] : slices["z"][index + 1]].long()
    pos = data.pos[slices["pos"][index] : slices["pos"][index + 1]].float()
    y = data.y[slices["y"][index] : slices["y"][index + 1]].view(-1).float()
    return z, pos, y


def _list_item(samples, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample = samples[index]
    z = sample["z"].view(-1).long()
    pos = sample["pos"].view(-1, 3).float()
    y = sample["y"].view(-1).float()
    return z, pos, y


def convert_qm9(
    input_path: Path,
    output_path: Path,
    target: str,
    max_molecules: int | None,
    extra_site_packages: Iterable[str],
) -> int:
    if target not in QM9_TARGETS:
        names = ", ".join(QM9_TARGETS)
        raise ValueError(f"Unknown target `{target}`. Available targets: {names}")

    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    payload = _load_qm9(input_path, extra_site_packages)
    data = None
    slices = None
    samples = None
    if isinstance(payload, list) and (len(payload) == 0 or isinstance(payload[0], dict)):
        samples = payload
    else:
        data, slices = _normalise_payload(payload)
    target_idx = QM9_TARGETS[target]
    num_molecules = len(samples) if samples is not None else int(slices["z"].numel() - 1)
    if max_molecules is not None:
        num_molecules = min(num_molecules, int(max_molecules))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    for index in range(num_molecules):
        if samples is None:
            z, pos, y = _slice_item(data, slices, index)
        else:
            z, pos, y = _list_item(samples, index)
        atoms = Atoms(numbers=z.numpy(), positions=pos.numpy())
        atoms.info["qm9_index"] = index
        atoms.info["qm9_target"] = target
        atoms.calc = SinglePointCalculator(atoms, energy=float(y[target_idx].item()))
        write(output_path, atoms, format="extxyz", append=index > 0)
        if (index + 1) % 10000 == 0:
            print(f"wrote {index + 1}/{num_molecules} molecules", flush=True)

    print(f"wrote {num_molecules} molecules to {output_path}", flush=True)
    return num_molecules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/chem_workspace/Chem_Graph_Transformer/data/qm9/raw/qm9_v3.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qm9/qm9_u0_atom.extxyz"),
    )
    parser.add_argument("--target", choices=sorted(QM9_TARGETS), default="u0_atom")
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument(
        "--extra-site-packages",
        action="append",
        default=[
            "/data/chem_workspace/Chem_Graph_Transformer/venv/lib/python3.12/site-packages"
        ],
        help="Extra site-packages path used to unpickle PyG QM9 objects.",
    )
    args = parser.parse_args()
    convert_qm9(
        input_path=args.input,
        output_path=args.output,
        target=args.target,
        max_molecules=args.max_molecules,
        extra_site_packages=args.extra_site_packages,
    )


if __name__ == "__main__":
    main()
