#!/usr/bin/env bash
set -euo pipefail

SRC="/Users/grantwilkins/powertrace-sim/results/training"
DEST="/Users/grantwilkins/powertrace-sim/model/random_gru_classifier_weights"
MODE="${MODE:-copy}"  # set MODE=move to move instead of copy

mkdir -p "$DEST"

src_base="$(basename "$SRC")"

# Find all .pt files (one per subfolder per your note; supports top-level too)
find "$SRC" -type f -name '*.pt' -print0 | while IFS= read -r -d '' ptfile; do
  parent="$(basename "$(dirname "$ptfile")")"
  if [[ "$parent" == "$src_base" ]]; then
    ref="$(basename "$ptfile")"
    ref="${ref%.pt}"
  else
    ref="$parent"
  fi

  # Parse {model}_{hardware}_{tp or number}
  last="${ref##*_}"             # tp part or number
  prev="${ref%_*}"
  hardware="${prev##*_}"
  model="${ref%_*_*}"

  if [[ "$last" =~ ^(tp)?([0-9]+)$ ]]; then
    tp="${BASH_REMATCH[2]}"
  else
    echo "WARN: cannot parse TP from '$ref'; skipping '$ptfile'"
    continue
  fi

  outname="${model}_${hardware}_tp${tp}.pt"
  outpath="${DEST}/${outname}"

  if [[ -e "$outpath" ]]; then
    echo "SKIP: $outpath already exists"
    continue
  fi

  echo "-> ${MODE}: '$ptfile' -> '$outpath'"
  if [[ "$MODE" == "move" ]]; then
    mv "$ptfile" "$outpath"
  else
    cp "$ptfile" "$outpath"
  fi
done