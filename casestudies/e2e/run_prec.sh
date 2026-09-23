#!/bin/bash
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:50:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J parsimony-prec
#SBATCH -o prec-%j.out
#
# Does fp32 storage buy anything end-to-end?
#
# The previous run reported 1.00x at 311k dof, but its per-event attribution was
# meaningless: PETSc's GPU MatMult is asynchronous, so the logged MatMult time is
# launch cost (0.4 us/call) and VecTDot -- the synchronisation point -- absorbs
# the real work.  The giveaway was VecTDot scaling linearly with N at an
# effective 10.6 GB/s, half a percent of A100 peak.
#
# -log_view_gpu_time forces per-event synchronisation so the attribution is
# honest.  KSPSolve wall time was always trustworthy (sync to sync) and remains
# the headline number; the paired plain/gputime runs show the profiling overhead.
#
# Sizes 750k and 1.48M dof, where the operator is large enough that bandwidth
# should dominate if it ever does.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
LOGDIR=prec-${SLURM_JOB_ID:-local}
mkdir -p "$LOGDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Verbatim from the ex56 kokkos "hot" testset, mesh size aside.
TEST="-dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1
      -petscspace_degree 2 -run_type 1 -max_conv_its 1 -n_solves 2
      -ksp_type cg -ksp_rtol 1.e-8 -ksp_norm_type unpreconditioned
      -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1
      -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true
      -use_mat_nearnullspace true -mg_levels_ksp_type chebyshev
      -mg_levels_pc_type pbjacobi -mg_coarse_pc_type pbjacobi
      -snes_type ksponly -snes_lag_jacobian -2 -ksp_converged_reason
      -dm_vec_type kokkos"

for F in 32 40; do
  for prec in double single; do
    for gt in plain gputime; do
      if [ "$gt" = gputime ]; then GT="-log_view_gpu_time"; else GT=""; fi
      tag="f$F-$prec-$gt"
      echo "@@@ $tag"
      if ! srun -n 1 "./ex56-$prec" $TEST -dm_plex_box_faces "$F,$F,$F" \
           -dm_mat_type aijkokkos $GT -log_view > "$LOGDIR/$tag.txt" 2>&1; then
        echo "  FAILED"
        grep -m3 "PETSC ERROR" "$LOGDIR/$tag.txt" | sed 's/^/  /'
        continue
      fi
      grep -m1 "Linear solve converged" "$LOGDIR/$tag.txt" | sed 's/^/  /'
      awk '/Event Stage 3: Hot Solve/,/^---/' "$LOGDIR/$tag.txt" \
        | grep -E '^(KSPSolve|MatMult|VecTDot|PCApply) ' \
        | awk '{printf "    %-9s calls=%-7s time=%s\n", $1, $2, $4}'
    done
  done
done
echo "@@@ done"
