#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${SRC_DIR:-/ucon/src}"
DST_DIR="${DST_DIR:-/omnigibson-src/omnigibson}"

if [ ! -d "$SRC_DIR" ]; then
  echo "[sync_ucon] ERROR: SRC_DIR not found: $SRC_DIR"
  exit 1
fi
mkdir -p "$DST_DIR"

echo "[sync_ucon] src: $SRC_DIR"
echo "[sync_ucon] dst: $DST_DIR"

tar -C "$SRC_DIR" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  -cf - . \
| tar -C "$DST_DIR" -xf -

echo "[sync_ucon] done"
