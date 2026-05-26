#!/usr/bin/env bash
set -euo pipefail

# Train one NequIP QM9 scalar-regression run per target.
#
# This follows the common Equiformer QM9 benchmark target set:
#   mu alpha homo lumo gap r2 zpve u0 u h g cv
#
# Override examples:
#   TARGETS="u0_atom u_atom h_atom g_atom" scripts/run_qm9_all_targets_repro.sh
#   PREPARE_ONLY=1 scripts/run_qm9_all_targets_repro.sh
#   FORCE_PREPARE=1 scripts/run_qm9_all_targets_repro.sh

PROJECT_ROOT="${PROJECT_ROOT:-/data/chem_workspace/nequip}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/.venv/bin/activate}"
QM9_INPUT="${QM9_INPUT:-/data/chem_workspace/Chem_Graph_Transformer/data/qm9/raw/qm9_v3.pt}"
EXTRA_SITE_PACKAGES="${EXTRA_SITE_PACKAGES:-/data/chem_workspace/Chem_Graph_Transformer/venv/lib/python3.12/site-packages}"
CONFIG_NAME="${CONFIG_NAME:-qm9_u0_atom_repro}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qm9_all_targets_repro}"
DATA_ROOT="${DATA_ROOT:-data/qm9}"
TARGETS="${TARGETS:-mu alpha homo lumo gap r2 zpve u0 u h g cv}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

cd "${PROJECT_ROOT}"
source "${VENV_ACTIVATE}"

mkdir -p "${DATA_ROOT}"

for target in ${TARGETS}; do
  extxyz_path="${DATA_ROOT}/qm9_${target}.extxyz"
  run_dir="${OUTPUT_ROOT}/${target}"

  if [[ "${FORCE_PREPARE}" == "1" || ! -s "${extxyz_path}" ]]; then
    python scripts/prepare_qm9_extxyz.py \
      --input "${QM9_INPUT}" \
      --output "${extxyz_path}" \
      --target "${target}" \
      --extra-site-packages "${EXTRA_SITE_PACKAGES}"
  fi

  if [[ "${PREPARE_ONLY}" == "1" ]]; then
    continue
  fi

  nequip-train \
    --config-path "${PROJECT_ROOT}/configs" \
    --config-name "${CONFIG_NAME}" \
    hydra.run.dir="${run_dir}" \
    data.split_dataset.file_path="${extxyz_path}"
done
