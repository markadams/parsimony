#!/bin/bash
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:55:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J parsimony-ex56k
#SBATCH -o ex56k-%j.out
#
# Restructured end-to-end study, on ex56k: 3D linear elasticity with Kokkos
# GPU-resident FEM assembly.  This exists only in the petsc-claude fork; upstream
# PETSc has no BAIJ-Kokkos and no Kokkos FE template path at all.
#
# Why ex56k rather than ex56.  It assembles the Jacobian ON THE DEVICE via block
# COO, so baijkokkos is built natively -- no -block_mat_type conversion -- and,
# more importantly, it exposes a second kernel class.  Assembly is compute-bound
# (quadrature and tensor contraction) where SpMV is bandwidth-bound, so the two
# should respond to reduced precision differently.  The earlier ex56 runs showed
# assembly costing ~12x the solve, so it is also the bigger target.
#
# Three arms: GPU assembly into scalar (aijkokkos) and block (baijkokkos), and a
# CPU-assembly control (aij) that must reproduce the same displacement.
#
# -log_view_gpu_time forces per-event synchronisation.  Without it PETSc's async
# GPU MatMult logs launch cost only (0.4 us/call) and the following VecTDot
# absorbs the real work, which made the previous sweep's per-event numbers
# meaningless.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
LOGDIR=ex56k-${SLURM_JOB_ID:-local}
mkdir -p "$LOGDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Solver held fixed across arms so the only difference is operator storage and
# where the Jacobian is assembled.  Follows the ex56k cube testset.
BASE="-dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1
      -petscspace_degree 2 -snes_max_it 1 -snes_type ksponly -n_solves 2
      -ksp_type cg -ksp_rtol 1.e-8 -ksp_norm_type unpreconditioned -ksp_max_it 500
      -pc_type gamg -pc_gamg_coarse_eq_limit 100 -mg_levels_ksp_max_it 2
      -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi
      -ksp_converged_reason"

run () {  # F prec armname args...
  local F=$1 prec=$2 arm=$3; shift 3
  local dof=$(( 3 * (2*F-1)**3 ))
  local tag="f$F-$prec-$arm"
  echo "@@@ $tag  dof=$dof"
  if ! srun -n 1 "./ex56k-$prec" $BASE -dm_plex_box_faces "$F,$F,$F" "$@" \
       -log_view_gpu_time -log_view > "$LOGDIR/$tag.txt" 2>&1; then
    echo "  FAILED"; grep -m4 "PETSC ERROR" "$LOGDIR/$tag.txt" | sed 's/^/  /'; return
  fi
  grep -m1 -iE "Max displacement" "$LOGDIR/$tag.txt" | sed 's/^/  /'
  grep -m1 "Linear solve converged" "$LOGDIR/$tag.txt" | sed 's/^/  /'
}

GPU_SCALAR="-dm_mat_type aijkokkos -dm_vec_type kokkos"
GPU_BLOCK="-dm_mat_type baijkokkos -dm_vec_type kokkos"
CPU="-dm_mat_type aij -dm_vec_type standard"

for prec in double single; do
  run 32 $prec gpu-scalar $GPU_SCALAR
  run 32 $prec gpu-block  $GPU_BLOCK
  run 32 $prec cpu        $CPU
done
for prec in double single; do
  run 40 $prec gpu-scalar $GPU_SCALAR
  run 40 $prec gpu-block  $GPU_BLOCK
done
echo "@@@ done"
