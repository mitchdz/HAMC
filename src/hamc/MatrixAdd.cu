#ifndef ADD_KERNEL_H
#define ADD_KERNEL_H

#include "hamc_cpu_code.c"
#include "hamc_common.h"

#define BLOCK_DIM 16
#define ADD_TILE_WIDTH 16

__global__ void MatrixAdd(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C,
        int rows, int cols);

bin_matrix run_kernel(bin_matrix A, bin_matrix B)
{
    if (A->rows != B->rows || A->cols != B->cols){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }
    
    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;
    HAMC_DATA_TYPE_t *deviceC;
    
    bin_matrix C = mat_init_cpu(A->rows,B->cols);

    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceC, B->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));

    
    cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
    
    printf("TILE_WIDTH -> %i \n", TILE_WIDTH);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((B->cols - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);
  
    MatrixAdd<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, A->rows, A->cols);

    //printf("\n");
    //printf("Total compute time (ms) %f for Matrix Add GPU\n\n",aelapsedTime);
    //printf("\n");
    

    cudaMemcpy(C->data, deviceC, B->cols * A->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
   
    return C;
}



__global__ void MatrixAdd(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C,
        int rows, int cols) {
        

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
	
    int index = row * cols + col;
 
    if((row < rows) && (col < cols)) {
  	    C[index] = A[index] ^ B[index];
    }

}
#endif // ADD_KERNEL_H
