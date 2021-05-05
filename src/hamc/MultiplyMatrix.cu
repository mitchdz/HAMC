/*
*
*
*
*
*
*/
#ifndef HAMC_MULTIPLY_MATRIX_H
#define HAMC_MULTIPLY_MATRIX_H

#include <stdio.h>
#include <cuda.h>
#include "hamc_common.h"

#define TILE_WIDTH 32

__global__ void mult_kernel_compressed_data(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB)
{
    //Shared variable created
    __shared__ uint32_t sharedFloatA[TILE_WIDTH * TILE_WIDTH];
    __shared__ uint32_t sharedFloatB[TILE_WIDTH * (TILE_WIDTH + 1)];
    //Shared variable typecast for later use
    HAMC_DATA_TYPE_t *sharedA = (uint8_t *)sharedFloatA;
    HAMC_DATA_TYPE_t *sharedB = (uint8_t *)sharedFloatB;
    
    //Typecasting of global variables for consolidated data accesses
    uint32_t *floatA = (uint32_t *)A;
    uint32_t *floatB = (uint32_t *)B;
    
    //Indecies definitions
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int tilePos = 0;
    
    //Local output variables
    uint32_t pValueFloat = 0;
    HAMC_DATA_TYPE_t shortValue = 0;
    
    //For loop for tiled access
    for(int i = 0; i < ((colA - 1)/(TILE_WIDTH * 4)) + 1; i++){
        //Tile indecies
        tilePos = i * TILE_WIDTH;
        //Boundary checking for global matrix A
        if((Row < rowA) && (tilePos + threadIdx.x < (colA - 1) / 4 + 1)){
            memcpy(&sharedFloatA[tid], &A[Row * colA + tilePos * 4 + threadIdx.x * 4], sizeof(uint32_t));
            //Further boundary conditions to handle boundaries inside the last global read
            if((tilePos + threadIdx.x) == colA / 4){
                int padding = colA % 4;
                for(int j = 3; j >= padding; j--){
                    sharedA[threadIdx.y * TILE_WIDTH * 4 + threadIdx.x * 4 + j] = (uint8_t)0;
                }
            }
        }
        else{
            sharedFloatA[tid] = (uint32_t)0;
        }
        //Matrix B still has to be accesed without consolidation
        for(int j = 0; j < 4; j++){
            #pragma unroll
            //Boundary checking for global matrix B
            if(((j * TILE_WIDTH + threadIdx.y + tilePos * 4) < rowB) && (Col < colB)){
                sharedB[threadIdx.x * 4 * (TILE_WIDTH + 1) + j * (TILE_WIDTH) + threadIdx.y] = B[colB * (j * TILE_WIDTH + threadIdx.y + tilePos * 4) + Col];
            }
            else{
                sharedB[threadIdx.x * 4 * (TILE_WIDTH + 1) + j * (TILE_WIDTH) + threadIdx.y] = (uint8_t)0;
            }
        }        
        __syncthreads();
        
        //Binary matrix multiplication performed for each tile
        for(int j = 0; j < TILE_WIDTH; j++){
            #pragma unroll
            pValueFloat ^= sharedFloatA[threadIdx.y * TILE_WIDTH + j] & sharedFloatB[threadIdx.x * (TILE_WIDTH + 1) + j];
        }
        __syncthreads();
    }
    
    //Output boundary conditions
    if(Row < rowA && Col < colB){
        //Final step of matrix multiplication
        for(int i = 0; i < 4; i++){
            #pragma unroll
            shortValue ^= pValueFloat & 1;
            pValueFloat >>= 8;
        }
        C[Row * colB + Col] = shortValue;
    }
}

bin_matrix run_mult_kernel(bin_matrix A, bin_matrix B)
{    
    if (A->cols != B->rows){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }

    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;
    HAMC_DATA_TYPE_t *deviceC;
    
    bin_matrix C = mat_init_cpu(A->rows, B->cols);
    
    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceC, B->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));
    
    cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
    
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((B->cols - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);
    
    mult_kernel_compressed_data<<<DimGrid, DimBlock, (2 * TILE_WIDTH * TILE_WIDTH + TILE_WIDTH) * sizeof(float)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols);
    
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
