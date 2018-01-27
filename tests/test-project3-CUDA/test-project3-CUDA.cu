// System includes --------------------
#include <iostream>

// Own includes --------------------
#include "project3-CUDA/project3-CUDA.cu"

int main( int argc, char* argv[] ) {

  std::size_t data_size = 100;
  dim3 block, grid;

  block = dim3(32);
  grid = dim3( ( data_size + block.x - 1 )/block.x );

  //kernel launch
  CUDATest<<< grid, block >>>( data_size );

  //error check and cleanup
  CUDA_CHECK( cudaPeekAtLastError() );
  CUDA_CHECK( cudaDeviceSynchronize() );

  return 0;
}
