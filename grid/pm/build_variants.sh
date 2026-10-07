#!/bin/bash
# Rebuild Grid-dev for the experiments, three builds in parallel on one login node:
#   build        incremental, picks up patch 0003 (GRID_DHOP_INTERIOR_FIRST)
#   build-rr64   fresh, nvcc -maxrregcount=64
#   build-rr80   fresh, nvcc -maxrregcount=80
# Library and Benchmark_dwf only. Each log ends in VARIANT_OK <name> on success.
set -uo pipefail
WORK=/pscratch/sd/m/madams/gridperf
SRC=/pscratch/sd/m/madams/Grid-dev
MPFR=/pscratch/sd/m/madams/sw/mpfr
module load PrgEnv-gnu gcc-native/14 cudatoolkit/13.2
export CRAY_ACCEL_TARGET=nvidia80
cd $SRC
git checkout -q -B madams/cuda13-build 3232b438
git -c user.name="Mark Adams" -c user.email=mfadams@lbl.gov am -q $WORK/patches/*.patch
git log --oneline -4 > $WORK/variants_src.txt
./bootstrap.sh > $WORK/variants_bootstrap.log 2>&1

conf () {   # $1 build dir, $2 extra nvcc flags
  mkdir -p $SRC/$1 && cd $SRC/$1
  ../configure --enable-comms=mpi --enable-simd=GPU --enable-shm=nvlink \
    --enable-gen-simd-width=64 --enable-accelerator=cuda --enable-tracing=nvtx \
    --disable-fermion-reps --disable-unified --disable-gparity --with-mpfr=$MPFR \
    CXX=nvcc LDFLAGS="-cudart shared -Xlinker -rpath -Xlinker $MPFR/lib" \
    CXXFLAGS="-ccbin CC -gencode arch=compute_80,code=sm_80 -std=c++17 -cudart shared $2"
}
variant () {   # $1 name, $2 extra nvcc flags ("" = incremental rebuild of build/)
  { date
    if [ "$1" = build ]; then
      cd $SRC/build && find Grid/qcd/action/fermion -name '*.o' -delete
    else
      conf $1 "$2" || exit 1
    fi
    make -j6 -C Grid && make -C benchmarks Benchmark_dwf Benchmark_dwf_fp32 && \
      ls -l benchmarks/Benchmark_dwf benchmarks/Benchmark_dwf_fp32 && date && echo "VARIANT_OK $1"
  } > $WORK/variant_$1.log 2>&1
}
variant build "" &
variant build-rr64 "-maxrregcount=64" &
variant build-rr80 "-maxrregcount=80" &
wait
echo ALL_DONE >> $WORK/variants_src.txt
