#!/usr/bin/env bash
set -euo pipefail

# Train one NequIP model on the combined rMD17 molecule set.
#
# The config default uses all 10 rMD17 molecules with a 95% / 2.5% / 2.5%
# train/validation/test split inside each molecule before concatenation.

PROJECT_ROOT="${PROJECT_ROOT:-/datadisk/chem_workspace/nequip}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/.venv/bin/activate}"
CONFIG_NAME="${CONFIG_NAME:-rmd17_all_molecules_repro}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rmd17_all_molecules_repro}"
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-data}"
DATASETS="${DATASETS:-aspirin azobenzene benzene ethanol malonaldehyde naphthalene paracetamol salicylic toluene uracil}"
TRAIN_VAL_TEST_SPLIT="${TRAIN_VAL_TEST_SPLIT:-[0.95,0.025,0.025]}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"

cd "${PROJECT_ROOT}"
source "${VENV_ACTIVATE}"

datasets_csv="${DATASETS// /,}"

nequip-train \
  --config-path "${PROJECT_ROOT}/configs" \
  --config-name "${CONFIG_NAME}" \
  "hydra.run.dir=${OUTPUT_DIR}" \
  "data.data_source_dir=${DATA_SOURCE_DIR}" \
  "data.datasets=[${datasets_csv}]" \
  "data.train_val_test_split=${TRAIN_VAL_TEST_SPLIT}" \
  "trainer.max_epochs=${MAX_EPOCHS}"
