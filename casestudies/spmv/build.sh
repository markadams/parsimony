#!/bin/bash
# Build the SpMV sweep against the Kokkos in $PETSC_DIR/$PETSC_ARCH.
#   PETSC_ARCH=arch-perlmutter-opt-gcc-kokkos-cuda ./build.sh
set -euo pipefail
: "${PETSC_DIR:?set PETSC_DIR}"
: "${PETSC_ARCH:?set PETSC_ARCH}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build-$PETSC_ARCH"

cmake -S "$HERE" -B "$BUILD" \
  -DCMAKE_PREFIX_PATH="$PETSC_DIR/$PETSC_ARCH" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j 16
echo "built: $BUILD/spmv"
