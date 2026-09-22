#!/bin/bash
# One GPU node-share on Perlmutter.  "shared" is charged at 1/4 rate and this
# needs a few minutes, so the whole tier-2 measurement costs well under a
# node-hour.  Submit with:  sbatch run_perlmutter.sh
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J parsimony-spmv
#SBATCH -o spmv-%j.out

set -euo pipefail
BUILD="${BUILD:-$SLURM_SUBMIT_DIR/build-$PETSC_ARCH}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun -n 1 "$BUILD/spmv" --reps 50

echo
echo "=== CPU (same kernel, host backend) ==="
echo "# Expect the mixed rows to underperform the byte model here: conversion"
echo "# happens in registers on the GPU but costs instructions on the CPU."
srun -n 1 "$BUILD/spmv" --reps 20 --small || true
