
#ifndef HAMC_MULTIPLY_MATRIX_H
#define HAMC_MULTIPLY_MATRIX_H

#include <stdio.h>
#include "hamc_common.h"

//#define TILE_WIDTH 16

//int TILE_WIDTH = 16;

__global__ void mult_kernel(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB, int TILE_WIDTH)
{
    extern __shared__ HAMC_DATA_TYPE_t sharedArray[];
    
    //int TILE_WIDTH = (sizeof(sharedArray) / sizeof(sharedArray[0])) / 4;
    
    HAMC_DATA_TYPE_t *sharedA = sharedArray;
    HAMC_DATA_TYPE_t *sharedB = &sharedA[TILE_WIDTH * TILE_WIDTH];
    //extern __shared__ HAMC_DATA_TYPE_t sharedA[];
    //extern __shared__ HAMC_DATA_TYPE_t sharedB[];
    
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int tilePos = 0;

    HAMC_DATA_TYPE_t pValue = 0;
  
    for(int i = 0; (i < ((colA - 1)/TILE_WIDTH) + 1) && (i < ((rowB - 1)/TILE_WIDTH) + 1); i++){
        tilePos = i * TILE_WIDTH;
        if((Row < rowA) && (tilePos + threadIdx.x < colA)){
            sharedA[tid] = A[Row * colA + tilePos + threadIdx.x];
        }
        else{
            sharedA[tid] = 0.0;
        }
        if((Col < colB) && (tilePos + threadIdx.y < rowB)){
            sharedB[tid] = B[(tilePos + threadIdx.y) * colB + Col];
        }
        else{
            sharedB[tid] = 0.0;
        }
        __syncthreads();
        
        if((Row < rowA) && (Col < colB)){
            for(int j = 0; j < TILE_WIDTH; j++){
                pValue ^= (sharedA[threadIdx.y * TILE_WIDTH + j] & sharedB[j * TILE_WIDTH + threadIdx.x]);
            }
        }
        
        __syncthreads();
    }
    if((Row < rowA) && (Col < colB)){
        C[Row * colB + Col] = pValue;
    }
}

bin_matrix run_mult_kernel(bin_matrix A, bin_matrix B)
{
    int TILE_WIDTH = 16;
    
    if (A->cols != B->rows){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }

    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;
    HAMC_DATA_TYPE_t *deviceC;
    
    bin_matrix C = mat_init_cpu(A->rows, A->cols);
    
    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceC, B->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));
    
    cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
    
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((B->cols - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);
    
    mult_kernel<<<DimGrid, DimBlock, TILE_WIDTH * TILE_WIDTH * sizeof(HAMC_DATA_TYPE_t)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols, TILE_WIDTH);
    
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    cudaMemcpy(C->data, deviceC, C->cols * C->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}

bin_matrix run_mult_kernel(bin_matrix A, bin_matrix B, int tile_width)
{
    int TILE_WIDTH = tile_width;
    
    if (A->cols != B->rows){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }

    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;
    HAMC_DATA_TYPE_t *deviceC;
    
    bin_matrix C = mat_init_cpu(A->rows, A->cols);
    
    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceC, B->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));
    
    cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
    
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((B->cols - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);
    
    mult_kernel<<<DimGrid, DimBlock, TILE_WIDTH * TILE_WIDTH * sizeof(HAMC_DATA_TYPE_t)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols, TILE_WIDTH);
    
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    cudaMemcpy(C->data, deviceC, C->cols * C->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}

#endif /* HAMC_MULTIPLY_MATRIX_H */
