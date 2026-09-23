#!/bin/bash
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:40:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J parsimony-e2e
#SBATCH -o e2e-%j.out
#
# End-to-end check of the cost model: does
#     time = iterations x cost-per-iteration
# predict wall time for a real PETSc solve rather than an extracted kernel?
#
# Arguments follow the ex56 "hot" testset (kokkos_kernels, suffixes hot_scalar /
# hot_block) rather than being invented here.  Two things that matters for:
#   * DMPlex does NOT build the Kokkos BAIJ structure directly.  The block run
#     assembles scalar MPIAIJ and converts with -block_mat_type mpibaijkokkos;
#     -dm_mat_type baijkokkos is the wrong path for this problem.
#   * -n_solves 2 runs a second solve reusing the GAMG hierarchy, logged in its
#     own "hot" stage.  Steady-state solve cost is read from that stage, not from
#     the first solve, which carries setup.
#
# Block math is identical to scalar math, so iteration counts should agree within
# a precision; any difference between single and double is the tier-1 effect.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
FACES=${FACES:-24}
LOGDIR=logs-$SLURM_JOB_ID
mkdir -p $LOGDIR

BASE="-dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0
      -dm_plex_box_upper 1,1,1 -dm_plex_box_faces $FACES,$FACES,$FACES
      -petscspace_degree 2 -run_type 1 -max_conv_its 1 -n_solves 2
      -ksp_type cg -ksp_rtol 1.e-8 -ksp_norm_type unpreconditioned
      -snes_type ksponly -snes_lag_jacobian -2 -ksp_converged_reason
      -dm_vec_type kokkos"

GAMG="-pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1
      -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true
      -use_mat_nearnullspace true -mg_levels_ksp_type chebyshev
      -mg_levels_pc_type pbjacobi -mg_coarse_pc_type pbjacobi"

# A single-level arm too: with point Jacobi the solve is MatMult-dominated, which
# is the cleanest test of the factorization.  GAMG is the realistic one.
JAC="-pc_type jacobi -ksp_max_it 2000"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "faces=$FACES degree=2"

run () {  # tag, executable, extra args...
  local tag=$1; shift
  local exe=$1; shift
  echo "@@@ $tag"
  srun -n 1 ./$exe $BASE "$@" -log_view :$LOGDIR/$tag.log:ascii_info_detail \
       > $LOGDIR/$tag.out 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED rc=$rc"; grep -m6 "PETSC ERROR" $LOGDIR/$tag.out | sed "s/^/  /"
  else
    grep -hE "Linear solve converged|N=" $LOGDIR/$tag.out | head -3 | sed "s/^/  /"
  fi
}

for prec in double single; do
  for pc in jac gamg; do
    [ $pc = jac ] && PCARGS="$JAC" || PCARGS="$GAMG"
    run $prec-scalar-$pc ex56-$prec $PCARGS -dm_mat_type aijkokkos
    run $prec-block-$pc  ex56-$prec $PCARGS -dm_mat_type mpiaij -block_mat_type mpibaijkokkos
  done
done
echo "@@@ done"
