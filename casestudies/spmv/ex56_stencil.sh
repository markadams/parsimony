#!/bin/bash
# Measure the block size and stencil of PETSc's snes/tutorials/ex56 (3D linear
# elasticity), so the bandwidth model uses the real matrix rather than a guess.
#
#   PETSC_DIR=... PETSC_ARCH=... ./ex56_stencil.sh
#
# Runs on a login node: each case does one Newton step with one CG iteration and
# only asks for matrix info.  Results (2026-09-22, arch-perlmutter-opt-gcc-
# kokkos-cuda-single) are in README.md; the block counts follow (3d-2)^3 for Q1
# and (4d-5)^3 for Q2 on a d^3 grid of interior nodes, so blocks per block row
# tends to 27 and 64.
set -euo pipefail
: "${PETSC_DIR:?set PETSC_DIR}"
: "${PETSC_ARCH:?set PETSC_ARCH}"

cd "$PETSC_DIR/src/snes/tutorials"
make ex56 >/dev/null

printf "%3s %5s %6s %4s %9s %9s %6s %10s %7s\n" \
       deg boxes rows bs nnz blockrows blocks blocks/row nnz/row
for deg in 1 2; do
  for nb in 4 8 12; do
    out=$(./ex56 -dm_plex_dim 3 -dm_plex_simplex 0 \
            -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 \
            -run_type 1 -dm_plex_box_faces $nb,$nb,$nb -max_conv_its 1 \
            -lx 1. -alpha .01 -petscspace_degree $deg \
            -ksp_type cg -ksp_max_it 1 -pc_type jacobi -snes_max_it 1 \
            -ksp_view_mat ::ascii_info -options_left 0 2>&1)
    rows=$(echo "$out" | grep -m1 -oP 'rows=\K[0-9]+')
    nnz=$(echo "$out" | grep -m1 -oP 'total: nonzeros=\K[0-9]+')
    [ -z "${rows:-}" ] && continue
    python3 -c "
rows, nnz, bs = $rows, $nnz, 3
br, blocks = rows // bs, nnz // (bs * bs)
print(f'{$deg:>3} {$nb:>5} {rows:>6} {bs:>4} {nnz:>9} {br:>9} {blocks:>6} "\
"{blocks/br:>10.1f} {nnz/rows:>7.1f}')"
  done
done
