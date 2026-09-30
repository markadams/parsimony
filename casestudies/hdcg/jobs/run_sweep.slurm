#!/bin/bash
#SBATCH -A m499_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -J bits-sweep
#SBATCH -o /pscratch/sd/m/madams/hdcg/sweep-%j.out
export MPICH_GPU_SUPPORT_ENABLED=1
cd /pscratch/sd/m/madams/hdcg
mkdir -p results
cat << ESC > select_gpu
#!/bin/bash
export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
exec numactl -m \$SLURM_LOCALID -N \$SLURM_LOCALID \$*
ESC
chmod +x ./select_gpu
BIN=/pscratch/sd/m/madams/Grid/build/tests/debug/Test_coarse_bits_nb20
BASE="--grid 16.16.16.32 --mpi 1.1.1.4 --accelerator-threads 8 --device-mem 8000 --gauge free --Ls 8"
# case 0 first and complete, so a timeout still leaves a whole ladder.
for C in 0 1; do
for FMT in v53 v40 v30 v24 v20 v16 v12 v10 v8 fp32 bf16 fp16; do
  echo "######### stencil=$FMT case=$C #########"
  srun -n 4 ./select_gpu $BIN $BASE --round-stencil $FMT --case $C \
       --json results/sw_${FMT}_c${C}.json 2>&1 | grep -E "PARSIMONY|Caught signal|out of memory|GRID_ASSERT"
done
done
echo "######### SWEEP DONE #########"
