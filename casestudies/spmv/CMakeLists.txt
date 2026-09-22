cmake_minimum_required(VERSION 3.16)
project(parsimony_spmv CXX)

# Kokkos is found from the PETSc arch that already provides it; point
# CMAKE_PREFIX_PATH at $PETSC_DIR/$PETSC_ARCH (build.sh does this).
find_package(Kokkos REQUIRED)

add_executable(spmv spmv.cpp)
target_link_libraries(spmv Kokkos::kokkos)
target_compile_features(spmv PRIVATE cxx_std_17)

# Host-only checks on the matrix and the bandwidth model; no Kokkos, no GPU.
# Run these before spending allocation:  ctest --test-dir build-$PETSC_ARCH
enable_testing()
add_executable(test_stencil test_stencil.cpp)
target_compile_features(test_stencil PRIVATE cxx_std_17)
add_test(NAME stencil COMMAND test_stencil)
