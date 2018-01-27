// System includes --------------------
#include <stdio.h> // for printf

// Own includes --------------------
#include "cuda-helpers.hpp"

//-------------------------------------------------------------------------
//Some Simple Test Kernels
__global__ void CUDATest( int n ) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i < n ) {
    printf("CUDATest: %i\n",i);
  }
}
