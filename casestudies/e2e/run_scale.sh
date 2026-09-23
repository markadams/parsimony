#!/bin/bash
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:40:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J parsimony-scale
#SBATCH -o scale-%j.out
#
# Step 1 of tracking down the VecTDot-dominates result: run the ex56 "hot_scalar"
# test AS PUBLISHED, only scaling the mesh, with plain text -log_view so the
# CpuToGpu / GpuToCpu transfer columns are visible.  Nothing of mine is added.
#
# Diagnostic: latency-bound events do not grow with N; bandwidth-bound ones do.
# Four sizes therefore separate the two without any further instrumentation.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
LOGDIR=scale-$SLURM_JOB_ID; mkdir -p $LOGDIR
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verbatim from the ex56 kokkos "hot" testset (suffix hot_scalar).
TEST="-dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1
      -petscspace_degree 2 -run_type 1 -max_conv_its 1 -n_solves 2
      -ksp_type cg -ksp_rtol 1.e-8 -ksp_norm_type unpreconditioned
      -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1
      -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true
      -use_mat_nearnullspace true -mg_levels_ksp_type chebyshev
      -mg_levels_pc_type pbjacobi -mg_coarse_pc_type pbjacobi
      -snes_type ksponly -snes_lag_jacobian -2 -ksp_converged_reason
      -dm_vec_type kokkos -dm_mat_type aijkokkos"

for F in 16 24 32 40; do
  dof=$(( 3 * (2*F-1)**3 ))
  echo "@@@ faces=$F  dof=$dof"
  srun -n 1 ./ex56-double $TEST -dm_plex_box_faces $F,$F,$F \
       -log_view > $LOGDIR/f$F.txt 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "  FAILED rc=$rc"; grep -m4 "PETSC ERROR" $LOGDIR/f$F.txt | sed "s/^/  /"; continue; fi
  grep -m1 "Linear solve converged" $LOGDIR/f$F.txt | sed "s/^/  /"
  # Hot Solve stage only: event, calls, time, and the GPU transfer columns.
  awk "/--- Event Stage 3: Hot Solve/,/^-----/" $LOGDIR/f$F.txt \
    | grep -E "^(MatMult|VecTDot|VecNorm|VecAXPY|KSPSolve|PCApply|VecDot) " \
    | awk "{printf \"    %-10s calls=%-7s time=%-10s CpuToGpu=%-6s GpuToCpu=%-6s\n\", \$1,\$2,\$3,\$(NF-4),\$(NF-2)}"
done
echo "@@@ done"
