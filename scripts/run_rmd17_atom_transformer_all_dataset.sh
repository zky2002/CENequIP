#!/usr/bin/env bash
set -euo pipefail

# rMD17 AtomTransformerEmbed experiment: train one model on the combined dataset.

PROJECT_ROOT="${PROJECT_ROOT:-/datadisk/chem_workspace/nequip}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/.venv/bin/activate}"
CONFIG_NAME="${CONFIG_NAME:-rmd17_all_molecules_repro}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rmd17_atom_transformer_all_molecules}"
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-data}"
DATASETS="${DATASETS:-aspirin azobenzene benzene ethanol malonaldehyde naphthalene paracetamol salicylic toluene uracil}"
TRAIN_VAL_TEST_SPLIT="${TRAIN_VAL_TEST_SPLIT:-[0.95,0.025,0.025]}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"

ATOM_Z_EMBEDDING_DIM="${ATOM_Z_EMBEDDING_DIM:-16}"
ATOM_RBF_DIM="${ATOM_RBF_DIM:-8}"
ATOM_TRANSFORMER_LAYERS="${ATOM_TRANSFORMER_LAYERS:-1}"
ATOM_TRANSFORMER_HEADS="${ATOM_TRANSFORMER_HEADS:-4}"
ATOM_DROPOUT="${ATOM_DROPOUT:-0.0}"

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
  "trainer.max_epochs=${MAX_EPOCHS}" \
  "training_module.model.atom_transformer_embed=true" \
  "training_module.model.atom_transformer_kwargs.z_embedding_dim=${ATOM_Z_EMBEDDING_DIM}" \
  "training_module.model.atom_transformer_kwargs.rbf_dim=${ATOM_RBF_DIM}" \
  "training_module.model.atom_transformer_kwargs.transformer_num_layers=${ATOM_TRANSFORMER_LAYERS}" \
  "training_module.model.atom_transformer_kwargs.transformer_num_heads=${ATOM_TRANSFORMER_HEADS}" \
  "training_module.model.atom_transformer_kwargs.dropout=${ATOM_DROPOUT}"
