#include <wb.h>

#include "hamc_cpu_code.c"
#include "hamc_common.h"

__global__ void MatrixAdd(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C,
        int rows, int cols) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
	
    int index = row * cols + col;
 
    if( (row < rows) && (col < cols) ) {
  	    C[index] = A[index] ^ B[index];
    }
}