NB=${1:-40}
cd /pscratch/sd/m/madams/Grid/build/tests/debug
S=/pscratch/sd/m/madams/Grid
M=/pscratch/sd/m/madams/sw/mpfr
set -x
nvcc -std=c++17 -x cu -DHAVE_CONFIG_H -DPARSIMONY_NBASIS=$NB -I. -I../../../tests/debug -I../../Grid -I$S -I$M/include \
  -O3 -ccbin CC -gencode arch=compute_80,code=sm_80 -std=c++17 -cudart shared \
  -Xcompiler -fno-strict-aliasing --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -fopenmp \
  -c -o Test_hdcg_parsimony_nb$NB.o ../../../tests/debug/Test_hdcg_parsimony.cc || exit 1
nvcc -std=c++17 -link -I$S -I$M/include -O3 -ccbin CC -gencode arch=compute_80,code=sm_80 -std=c++17 \
  -cudart shared -Xcompiler -fno-strict-aliasing --expt-extended-lambda --expt-relaxed-constexpr \
  -Xcompiler -fopenmp -L$S/build/Grid -L$M/lib -cudart shared -Xlinker -rpath -Xlinker $M/lib \
  -Xcompiler -fopenmp -o Test_hdcg_parsimony_nb$NB Test_hdcg_parsimony_nb$NB.o ../../Grid/libGrid.a \
  -lcrypto -lunwind-x86_64 -lunwind -lmpfr -lgmp -lstdc++ -lm -lcuda -lcublas -lcufft -lz || exit 1
echo DRIVER_BUILD_OK nbasis=$NB
