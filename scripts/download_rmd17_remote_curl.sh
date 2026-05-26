#!/usr/bin/env bash
set -euo pipefail

URL='https://archive.materialscloud.org/record/file?filename=rmd17.tar.bz2&record_id=466'
ROOT="${1:-/datadisk/chem_workspace/nequip}"
OUT="$ROOT/data/rmd17.tar.bz2"
TMP="$OUT.part"

mkdir -p "$ROOT/data"
rm -f "$OUT"

curl \
  --fail \
  --location \
  --retry 5 \
  --retry-delay 5 \
  --continue-at - \
  --output "$TMP" \
  "$URL"

mv "$TMP" "$OUT"
ls -lh "$OUT"
