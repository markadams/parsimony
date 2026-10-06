#!/bin/bash
# Build Grid develop 3232b438 + grid/patches on a Perlmutter login node:
# CUDA 13.2, sm_80, NVTX tracing.  Full log in $WORK/build.log; success is
# the line BUILD_OK at its end, never the absence of an error.
#   nohup bash build_grid.sh &
set -euo pipefail
WORK=/pscratch/sd/m/madams/gridperf
SRC=/pscratch/sd/m/madams/Grid-dev
MPFR=/pscratch/sd/m/madams/sw/mpfr
exec > $WORK/build.log 2>&1
date
module load PrgEnv-gnu gcc-native/14 cudatoolkit/13.2
module list
export CRAY_ACCEL_TARGET=nvidia80

[ -d $SRC/.git ] || git clone -b develop https://github.com/paboyle/Grid.git $SRC
cd $SRC
git fetch -q origin
git checkout -q -B madams/cuda13-build 3232b438
git -c user.name="Mark Adams" -c user.email=mfadams@lbl.gov am $WORK/patches/*.patch
git log --oneline -3
./bootstrap.sh

mkdir -p build && cd build
../configure --enable-comms=mpi --enable-simd=GPU --enable-shm=nvlink \
  --enable-gen-simd-width=64 --enable-accelerator=cuda --enable-tracing=nvtx \
  --disable-fermion-reps --disable-unified --disable-gparity \
  --with-mpfr=$MPFR \
  CXX=nvcc LDFLAGS="-cudart shared -Xlinker -rpath -Xlinker $MPFR/lib" \
  CXXFLAGS="-ccbin CC -gencode arch=compute_80,code=sm_80 -std=c++17 -cudart shared"
grep -n "GRID_TRACING" Grid/Config.h
make -j8
make -C benchmarks Benchmark_dwf
ls -l benchmarks/Benchmark_dwf
date
echo BUILD_OK
