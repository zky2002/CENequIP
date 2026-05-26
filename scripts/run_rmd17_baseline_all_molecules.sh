#!/usr/bin/env bash
set -euo pipefail

# Reproduce the rMD17 baseline protocol: train one model per molecule.
#
# Default protocol:
#   950 train / 50 validation / remaining test frames per molecule.

PROJECT_ROOT="${PROJECT_ROOT:-/datadisk/chem_workspace/nequip}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/.venv/bin/activate}"
CONFIG_NAME="${CONFIG_NAME:-rmd17_aspirin_repro}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rmd17_baseline_by_molecule}"
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-data}"
TARGETS="${TARGETS:-aspirin azobenzene benzene ethanol malonaldehyde naphthalene paracetamol salicylic toluene uracil}"

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
    "chemical_species=[${species}]"
done
