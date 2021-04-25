#include <wb.h>

#include "hamc_cpu_code.c"
#include "hamc_common.h"

__global__ void MatrixAdd(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C,
        int rows, int cols) {
        
    // THIS WORKS
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Naive Implementation
    printf("A\n");	
    int index = row * cols + col;
 
    if((row < rows) && (col < cols)) {
  	    C[index] = A[index] ^ B[index];
    }
    
//    // Strided Access
//    printf("B\n");
//    int strideX = blockDim.x*gridDim.x;
//    int strideY = blockDim.y*gridDim.y;
//    //printf("tid -> %i\n" , i);
//    while(row < rows && col < cols)
//    {
//        C[row*rows + col] = A[row*rows + col] ^ B[row*rows + col];
//        row += strideX;
//        col += strideY;
//    }
    

}
