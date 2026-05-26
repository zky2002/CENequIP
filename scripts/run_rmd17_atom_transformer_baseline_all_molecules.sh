#!/usr/bin/env bash
set -euo pipefail

# rMD17 AtomTransformerEmbed experiments: train one model per molecule.
# Keeps the baseline 950 / 50 / remaining split from rmd17_aspirin_repro.

PROJECT_ROOT="${PROJECT_ROOT:-/datadisk/chem_workspace/nequip}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/.venv/bin/activate}"
CONFIG_NAME="${CONFIG_NAME:-rmd17_aspirin_repro}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rmd17_atom_transformer_by_molecule}"
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-data}"
TARGETS="${TARGETS:-aspirin azobenzene benzene ethanol malonaldehyde naphthalene paracetamol salicylic toluene uracil}"

ATOM_Z_EMBEDDING_DIM="${ATOM_Z_EMBEDDING_DIM:-16}"
ATOM_RBF_DIM="${ATOM_RBF_DIM:-8}"
ATOM_TRANSFORMER_LAYERS="${ATOM_TRANSFORMER_LAYERS:-1}"
ATOM_TRANSFORMER_HEADS="${ATOM_TRANSFORMER_HEADS:-4}"
ATOM_DROPOUT="${ATOM_DROPOUT:-0.0}"

cd "${PROJECT_ROOT}"
source "${VENV_ACTIVATE}"

species_for_molecule() {
  case "$1" in
    aspirin|ethanol|malonaldehyde|salicylic)
      printf 'C,H,O'
      ;;
    azobenzene)
      printf 'C,H,N'
      ;;
    benzene|naphthalene|toluene)
      printf 'C,H'
      ;;
    paracetamol|uracil)
      printf 'C,H,N,O'
      ;;
    *)
      echo "Unknown rMD17 molecule: $1" >&2
      exit 2
      ;;
  esac
}

test_size_for_molecule() {
  local molecule="$1"
  local npz_path="${DATA_SOURCE_DIR}/rmd17/npz_data/rmd17_${molecule}.npz"
  python - "${npz_path}" <<'PY'
from pathlib import Path
import sys
import numpy as np

path = Path(sys.argv[1])
with np.load(path, mmap_mode="r") as data:
    num_frames = int(data["energies"].shape[0])
test_size = num_frames - 950 - 50
if test_size <= 0:
    raise ValueError(f"Not enough frames in {path}: {num_frames}")
print(test_size)
PY
}

for molecule in ${TARGETS}; do
  species="$(species_for_molecule "${molecule}")"
  test_size="$(test_size_for_molecule "${molecule}")"
  nequip-train \
    --config-path "${PROJECT_ROOT}/configs" \
    --config-name "${CONFIG_NAME}" \
    "hydra.run.dir=${OUTPUT_ROOT}/${molecule}" \
    "data.dataset=${molecule}" \
    "data.data_source_dir=${DATA_SOURCE_DIR}" \
    "data.train_val_test_split=[950,50,${test_size}]" \
    "model_type_names=[${species}]" \
    "chemical_species=[${species}]" \
    "training_module.model.atom_transformer_embed=true" \
    "training_module.model.atom_transformer_kwargs.z_embedding_dim=${ATOM_Z_EMBEDDING_DIM}" \
    "training_module.model.atom_transformer_kwargs.rbf_dim=${ATOM_RBF_DIM}" \
    "training_module.model.atom_transformer_kwargs.transformer_num_layers=${ATOM_TRANSFORMER_LAYERS}" \
    "training_module.model.atom_transformer_kwargs.transformer_num_heads=${ATOM_TRANSFORMER_HEADS}" \
    "training_module.model.atom_transformer_kwargs.dropout=${ATOM_DROPOUT}"
done
