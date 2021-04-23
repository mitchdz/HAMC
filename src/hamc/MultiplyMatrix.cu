
#ifndef HAMC_MULTIPLY_MATRIX_H
#define HAMC_MULTIPLY_MATRIX_H

#include <stdio.h>
#include <cuda.h>
//#include <cuda/pipeline>
#include "hamc_common.h"

//#define TILE_WIDTH 16

//int TILE_WIDTH = 16;

/*__global__ void mult_kernel_outer_product(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB, int TILE_WIDTH)
{
    extern __shared__ HAMC_DATA_TYPE_t sharedA[];
    
    //int TILE_WIDTH = (sizeof(sharedArray) / sizeof(sharedArray[0])) / 4;
    
    //HAMC_DATA_TYPE_t *sharedA = sharedArray;
    //HAMC_DATA_TYPE_t *sharedB = &sharedA[TILE_WIDTH * TILE_WIDTH];
    //extern __shared__ HAMC_DATA_TYPE_t sharedA[];
    //extern __shared__ HAMC_DATA_TYPE_t sharedB[];
    
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int tilePos = 0;

    HAMC_DATA_TYPE_t b = 0;
    HAMC_DATA_TYPE_t pValue = 0;
  
    for(int i = 0; (i < ((colA - 1)/TILE_WIDTH) + 1) && (i < ((rowB - 1)/TILE_WIDTH) + 1); i++){
        tilePos = i * TILE_WIDTH;
        if((Row < rowA) && (tilePos + threadIdx.x < colA)){
            sharedA[tid] = A[Row * colA + tilePos + threadIdx.x];
        }
        else{
            sharedA[tid] = 0;
        }
        if((Col < colB) && (tilePos + threadIdx.y < rowB)){
            b = B[(tilePos + threadIdx.y) * colB + Col];
        }
        else{
            b = 0;
        }
        __syncthreads();
        
        if((Row < rowA) && (Col < colB)){
            for(int j = 0; j < TILE_WIDTH; j++){
                b = B[(j * colB) +  * (blockDim.x * blockDim.y)];
                pValue ^= (sharedA[threadIdx.y * TILE_WIDTH + j] & b);
            }
        }
        
        __syncthreads();
    }
    if((Row < rowA) && (Col < colB)){
        C[Row * colB + Col] = pValue;
    }
}/**/

/*__global__ void mult_kernel_register_blocked(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB){
    __shared__ HAMC_DATA_TYPE_t sharedA[64*64];
    __shared__ HAMC_DATA_TYPE_t sharedB[64*64];
    
    int tile = 64;
    HAMC_DATA_TYPE_t regC[16];
    
    int tileRow = blockIdx.y * tile;
    int tileCol = blockIdx.x * tile;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;//0-255
    int tilePosX = tid % tile;//0-63
    int tilePosY = tid / tile;//0-3
    int row = 0, sharedIndex = 0;
    int stride = (blockDim.x * blockDim.y) / tile;
    for(int i = 0; (i < ((colA - 1)/tile) + 1) && (i < ((rowB - 1)/tile) + 1) ; i++){
        for(int j = 0; j < (tile * tile) / (blockDim.x * blockDim.y); j++){//0-15
            row = j * stride + tilePosY;
            sharedIndex = row * tile + tilePosX;
            if(((tileRow + row) < rowA) && (((i * tile) + tilePosX) < colA)){
                sharedA[sharedIndex] = A[((tileRow + row) * colA) + (i * tile) + tilePosX];
            }
            else{
                sharedA[sharedIndex] = 0;
            }
            if(((row + (i * tile)) < rowB) && ((tileCol + tilePosX) < colB)){
                sharedB[sharedIndex] = B[(colB * (row + (i * tile))) + tileCol + tilePosX];
            }
            else{
                sharedB[sharedIndex] = 0;
            }
        }
        __syncthreads();
        for(int j = 0; j < blockDim.x; j++){
            //stride = j * blockDim.x / tile;
            for(int k = 0; k < tile; k++){
                regC[j] ^= sharedA[((j * blockDim.x / tile) + (tid % blockDim.x)) * tile + k] & sharedB[(k * tile) + (j * blockDim.x / tile) + (tid % blockDim.x)];
            }
        }
        __syncthreads();
    }
    __syncthreads();
    for(int i = 0; i < blockDim.x; i++){
        C[(tileRow + tid % blockDim.x + blockDim.y * (i / stride) * colB) + tileCol + tid % tile] = regC[i];
    }
}/**/

__global__ void mult_kernel_compressed_data(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB, int TILE_WIDTH)
{
    extern __shared__ HAMC_DATA_TYPE_t sharedArray[];
    
    HAMC_DATA_TYPE_t *sharedA = sharedArray;
    uint32_t *sharedFloatA = (uint32_t *)sharedA;
    HAMC_DATA_TYPE_t *sharedB = &sharedA[TILE_WIDTH * TILE_WIDTH];
    uint32_t *sharedFloatB = (uint32_t *)sharedB;
    
    uint32_t *floatA = (uint32_t *)A;
    
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int tilePos = 0;

    uint32_t pValue = 0;
    HAMC_DATA_TYPE_t shortValue = 0;
    
    for(int i = 0; i < ((colA - 1)/(TILE_WIDTH / 4)) + 1; i++){
        tilePos = i * TILE_WIDTH;
        sharedFloatA[tid] = floatA[Row * colA + tilePos + threadIdx.x];
        for(int j = 0; j < 4; j++){
            sharedB[tid * 4 + j] = B[(j + ((tilePos + threadIdx.y) * 4)) * colB + Col];
        }
        __syncthreads();
        for(int j = 0; j < TILE_WIDTH; j++){
            pValue ^= (uint32_t)sharedFloatA[threadIdx.y * TILE_WIDTH + j] & (uint32_t)sharedFloatB[j * TILE_WIDTH + threadIdx.x];
        }
    }
    //TODO: xor all pValue bits
    for(int i = 0; i < 4; i++){
        shortValue ^= pValue & 1;
        pValue >>= 8;
    }
    C[Row * colB + Col] = pValue;
}/**/

/*__global__ void mult_kernel_compressed_data(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB, int TILE_WIDTH)
{
    extern __shared__ HAMC_DATA_TYPE_t sharedArray[];
    
    HAMC_DATA_TYPE_t *sharedA = sharedArray;
    uint32_t *sharedFloatA = (uint32_t *)sharedA;
    uint32_t *sharedFloatB = &sharedFloatA[TILE_WIDTH * TILE_WIDTH];
    HAMC_DATA_TYPE_t *sharedB = (HAMC_DATA_TYPE_t *)sharedFloatB;
    
    uint8_t tempB[4];
    uint32_t *tempFloatB = (uint32_t *)tempB;
    
    uint8_t boundaryA[4];
    uint32_t *boundaryFloatA = (uint32_t *)boundaryA;
    uint8_t boundaryB[4];
    uint32_t *boundaryFloatB = (uint32_t *)boundaryB;
    
    uint32_t *floatA = (uint32_t *)A;
    uint32_t *floatB = (uint32_t *)B;
    
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int tilePos = 0;

    uint32_t pValue = 0;
    HAMC_DATA_TYPE_t shortValue = 0;
    
    for(int i = 0; i < ((colA - 1)/(TILE_WIDTH / 4)) + 1; i++){
        tilePos = i * TILE_WIDTH;
        if((Row < rowA) && (tilePos + threadIdx.x < colA / 4)){
            sharedFloatA[tid] = floatA[Row * colA + tilePos + threadIdx.x];
        }
        else if(Row >= rowA){
            sharedFloatA[tid] = 0;
        }
        else{
            boundaryFloatA[0] = floatA[Row * colA + tilePos + threadIdx.x];
            for(int j = 0; j < colA % 4; j++){
                boundaryA[3 - j] &= (uint8_t)0;
            }
            sharedFloatA[tid] = boundaryFloatA[0];
        }
        if((Col < colB) && (tilePos + threadIdx.y < rowB)){
            tempFloatB[0] = floatB[((threadIdx.x / 8) + ((threadIdx.y + tilePos) * 4)) * colB + (blockIdx.x * TILE_WIDTH / 8) + (threadIdx.x % 8)];
        }
        else if((tilePos + threadIdx.y) * 4 >= rowB){
            tempFloatB[0] = 0;
        }
        else{
            boundaryFloatB[0] = floatB[((threadIdx.x / 8) + ((threadIdx.y + tilePos) * 4)) * colB + (blockIdx.x * TILE_WIDTH / 8) + (threadIdx.x % 8)];
            for(int j = 0; j < colB % 4; j++){
                boundaryB[3 - j] &= (uint8_t)0;
            }
            tempFloatB[0] = boundaryFloatB[0];
        }
        #pragma unroll
        for(int j = 0; j < 4; j++){
            sharedB[(j + threadIdx.y) * TILE_WIDTH + threadIdx.x] = tempB[j];
            //sharedB[tid * 4 + j] = B[(j + ((tilePos + threadIdx.y) * 4)) * colB + Col];
        }
        __syncthreads();
        for(int j = 0; j < TILE_WIDTH; j++){
            pValue ^= sharedFloatA[threadIdx.y * TILE_WIDTH + j] & sharedFloatB[j * TILE_WIDTH + threadIdx.x];
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i = 0; i < 4; i++){
        shortValue ^= pValue & 1;
        pValue >>= 8;
    }
    __syncthreads();
    C[Row * colB + Col] = shortValue;
}/**/

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
            sharedA[tid] = 0;
        }
        if((Col < colB) && (tilePos + threadIdx.y < rowB)){
            sharedB[tid] = B[(tilePos + threadIdx.y) * colB + Col];
        }
        else{
            sharedB[tid] = 0;
        }
        __syncthreads();
        
        if((Row < rowA) && (Col < colB)){
            for(int j = 0; j < TILE_WIDTH; j++){
                pValue ^= (sharedA[threadIdx.y * TILE_WIDTH + j] & sharedB[j * TILE_WIDTH + threadIdx.x]);
            }
        }
        
        __syncthreads();
    }/**/
    /*for(int i = 0; (i < ((colA - 1)/TILE_WIDTH) + 1); i++){
        tilePos = i * TILE_WIDTH;
        if((Row < rowA) && (tilePos + threadIdx.x < colA)){
            sharedA[tid] = A[Row * colA + tilePos + threadIdx.x];
        }
        else{
            sharedA[tid] = 0;
        }
        for(int k = 0; k < ((rowB - 1)/TILE_WIDTH) + 1; k++){
            if((Col < colB) && (tilePos + threadIdx.y < rowB)){
                sharedB[tid] = B[(tilePos + threadIdx.y) * colB + Col];
            }
            else{
                sharedB[tid] = 0;
            }
            __syncthreads();
            
            if((Row < rowA) && (Col < colB)){
                for(int j = 0; j < TILE_WIDTH; j++){
                    pValue ^= (sharedA[threadIdx.y * TILE_WIDTH + j] & sharedB[j * TILE_WIDTH + threadIdx.x]);
                }
            }
            
            __syncthreads();
        }
    }/**/
    if((Row < rowA) && (Col < colB)){
        C[Row * colB + Col] = pValue;
    }
}

/*__global__ void mult_kernel_async(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB, int TILE_WIDTH)
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
            //sharedA[tid] = A[Row * colA + tilePos + threadIdx.x];
            __pipeline_memcpy_asyc(&sharedA[tid], &A[Row * colA + tilePos + threadIdx.x], sizeof(HAMC_DATA_TYPE_t));
        }
        else{
            sharedA[tid] = 0;
        }
        if((Col < colB) && (tilePos + threadIdx.y < rowB)){
            //sharedB[tid] = B[(tilePos + threadIdx.y) * colB + Col];
            __pipeline_memcpy_asyc(&sharedB[tid], &B[(tilePos + threadIdx.y) * colB + Col], sizeof(HAMC_DATA_TYPE_t));
        }
        else{
            sharedB[tid] = 0;
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);

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
}*/

bin_matrix run_mult_kernel(bin_matrix A, bin_matrix B)
{
    int TILE_WIDTH = 32;
    
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
    
    mult_kernel<<<DimGrid, DimBlock, 2 * TILE_WIDTH * TILE_WIDTH * sizeof(HAMC_DATA_TYPE_t)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols, TILE_WIDTH);
    
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    cudaMemcpy(C->data, deviceC, C->cols * C->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    //printf("C Row: %d, C Col: %d\n", C->rows)
    return C;
}

bin_matrix run_mult_kernel(bin_matrix A, bin_matrix B, int TILE_WIDTH)
{
    //int TILE_WIDTH = tile_width;
    
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
    
    mult_kernel<<<DimGrid, DimBlock, 2 * TILE_WIDTH * TILE_WIDTH * sizeof(HAMC_DATA_TYPE_t)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols, TILE_WIDTH);
    
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    cudaMemcpy(C->data, deviceC, C->cols * C->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}

bin_matrix run_mult_kernel_test(bin_matrix A, bin_matrix B, int TILE_WIDTH)
{
    //int TILE_WIDTH = tile_width;
    
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
    
    mult_kernel_compressed_data<<<DimGrid, DimBlock, 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols, TILE_WIDTH);
    //mult_kernel_outer_product<<<DimGrid, DimBlock, TILE_WIDTH * TILE_WIDTH * sizeof(HAMC_DATA_TYPE_t)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols, TILE_WIDTH);
    
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
