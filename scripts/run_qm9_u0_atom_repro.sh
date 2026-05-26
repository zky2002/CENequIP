#!/usr/bin/env bash
set -euo pipefail

cd /data/chem_workspace/nequip
source .venv/bin/activate

nequip-train \
  --config-path /data/chem_workspace/nequip/configs \
  --config-name qm9_u0_atom_repro \
  hydra.run.dir=outputs/qm9_u0_atom_repro
