#!/bin/sh
# Reconstructs the ELDiR-derived simulator files. See patches/README.md.
set -e
ELDIR_DIR="${TMPDIR:-/tmp}/eldir-divcoll"
BASELINE=bcd88b08
REPO="$(cd "$(dirname "$0")" && pwd)"

echo "Cloning ELDiR..."
rm -rf "$ELDIR_DIR"
git clone --quiet https://github.com/lstrgar/ELDiR.git "$ELDIR_DIR"
cd "$ELDIR_DIR"
git checkout --quiet "$BASELINE"

echo "Applying DivColl modifications..."
git apply "$REPO/patches/eldir-to-divcoll.patch"

echo "Installing simulator files..."
cd "$REPO"
mkdir -p simulator operators/defaults utils
cp "$ELDIR_DIR/simulator/sim.py"                 simulator/
cp "$ELDIR_DIR/simulator/utils.py"               simulator/
cp "$ELDIR_DIR/operators/defaults/geno_pheno.py" operators/defaults/
cp "$ELDIR_DIR/operators/defaults/mutate.py"     operators/defaults/
cp "$ELDIR_DIR/operators/defaults/select.py"     operators/defaults/
cp "$ELDIR_DIR/utils/disk_utils.py"              utils/
cp "$ELDIR_DIR/utils/optim_utils.py"             utils/

echo "Done. Simulator reconstructed."
