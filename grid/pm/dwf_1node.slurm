#!/bin/bash
#SBATCH -A m1489_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -J dwf1n
#SBATCH -o /pscratch/sd/m/madams/gridperf/runs/%x-%j.out
#
# Benchmark_dwf at Peter's production per-GPU volume (16x8x12x24, Ls 24; his
# 48^3x96 on --mpi 3.6.4.4) on one Perlmutter node, 4 A100s.
#   sbatch --export=ALL,MODE=base dwf_1node.slurm   timing only
#   sbatch --export=ALL,MODE=nsys dwf_1node.slurm   timeline, all ranks
#   sbatch --export=ALL,MODE=ncu  dwf_1node.slurm   counters, rank 0, Dslash ranges
#   optional: BINNAME=Benchmark_dwf_fp32  EXTRA="--dslash-unroll"  TAG=_unroll (run dir suffix)
#   optional: BUILDDIR=build-rr64 (build-rr80); any env such as GRID_DHOP_INTERIOR_FIRST=1
#   MODE=cells: every variant in CELLS below, twice each, back to back on one node
# Everything is kept in $RUN; nothing is filtered here.
set -u
MODE=${MODE:-base}
WORK=/pscratch/sd/m/madams/gridperf
BUILD=/pscratch/sd/m/madams/Grid-dev/${BUILDDIR:-build}
BIN=$BUILD/benchmarks/${BINNAME:-Benchmark_dwf}
RUN=$WORK/runs/dwf_${MODE}${TAG:-}_${SLURM_JOB_ID}
mkdir -p $RUN && cd $RUN

module load PrgEnv-gnu gcc-native/14 cudatoolkit/13.2
export SLURM_CPU_BIND="cores"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_GPU_IPC_ENABLED=1
export MPICH_GPU_EAGER_REGISTER_HOST_MEM=0
export MPICH_GPU_NO_ASYNC_MEMCPY=0

{ echo "MODE=$MODE JOB=$SLURM_JOB_ID NODES=$SLURM_JOB_NODELIST"; date
  git -C /pscratch/sd/m/madams/Grid-dev log --oneline -3
  module list 2>&1; nvidia-smi -L; ncu --version; nsys --version; } > env.txt 2>&1

cat << EOF > select_gpu
#!/bin/bash
export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
exec numactl -m \$SLURM_LOCALID -N \$SLURM_LOCALID \$*
EOF
chmod +x select_gpu

cat << 'EOF' > ncu_rank0
#!/bin/bash
if [[ ${SLURM_PROCID} == "0" ]]; then
  exec ncu --nvtx --print-nvtx-rename kernel \
    --nvtx-include "DhopInterior/" --nvtx-include "DhopExterior/" \
    --nvtx-include "Gather/" --nvtx-include "Merge/" \
    --launch-skip 8 --launch-count 16 --set full \
    -o ncu_rank0 "$@"
else
  exec "$@"
fi
EOF
chmod +x ncu_rank0

ARGS="--grid 16.8.24.48 --mpi 1.1.2.2 -Ls 24 --accelerator-threads 8 --shm 2048 --comms-overlap --shm-mpi 0 ${EXTRA:-}"
echo "ARGS=$ARGS" >> env.txt

case $MODE in
  base)
    srun ./select_gpu $BIN $ARGS > dwf.log 2>&1 ;;
  nsys)
    srun ./select_gpu nsys profile -t cuda,nvtx -o nsys.rank%q{SLURM_PROCID} \
         $BIN $ARGS > dwf.log 2>&1
    nsys stats -r nvtx_sum,cuda_gpu_kern_sum --format csv -o nsys0 nsys.rank0.nsys-rep > nsys_stats.log 2>&1 ;;
  cells)
    CELLS="base:build: ifirst:build:GRID_DHOP_INTERIOR_FIRST=1 rr64:build-rr64: rr80:build-rr80: rr64_ifirst:build-rr64:GRID_DHOP_INTERIOR_FIRST=1 unroll_ifirst:build:GRID_DHOP_INTERIOR_FIRST=1:--dslash-unroll"
    for rep in 1 2; do
      for cell in $CELLS; do
        IFS=: read name bdir envs xargs <<< "$cell"
        echo "cell $name rep $rep build $bdir env '$envs' args '$xargs'" >> env.txt
        env $envs srun ./select_gpu /pscratch/sd/m/madams/Grid-dev/$bdir/benchmarks/${BINNAME:-Benchmark_dwf} $ARGS $xargs > cell_${name}_r${rep}.log 2>&1
      done
    done ;;
  ncu)
    srun --ntasks-per-node=1 dcgmi profile --pause
    srun ./select_gpu ./ncu_rank0 $BIN $ARGS > dwf.log 2>&1
    srun --ntasks-per-node=1 dcgmi profile --resume
    ncu --import ncu_rank0.ncu-rep --csv --page raw --print-units base > ncu_raw.csv 2> ncu_export.err
    ncu --import ncu_rank0.ncu-rep --csv --page details --print-units base > ncu_details.csv 2>> ncu_export.err ;;
esac
echo "exit=$? $(date)" >> env.txt
ls -l >> env.txt
