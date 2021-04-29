
#ifndef HAMC_MULTIPLY_MATRIX_H
#define HAMC_MULTIPLY_MATRIX_H

#include <stdio.h>
#include <cuda.h>
//#include <cuda/pipeline>
#include "hamc_common.h"

#define TILE_WIDTH 32

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

/*__global__ void mult_kernel_register_blocked(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB)
{
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

//__global__ void mult_kernel_compressed_data(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB, int TILE_WIDTH)
//{
//    extern __shared__ HAMC_DATA_TYPE_t sharedArray[];
//    
//    HAMC_DATA_TYPE_t *sharedA = sharedArray;
//    uint32_t *sharedFloatA = (uint32_t *)sharedA;
//    uint32_t *sharedFloatB = &sharedFloatA[TILE_WIDTH * TILE_WIDTH];
//    HAMC_DATA_TYPE_t *sharedB = (uint8_t *)sharedFloatB;
//    
//    uint32_t *floatA = (uint32_t *)A;
//    uint32_t *floatB = (uint32_t *)B;
//    
//    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
//    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
//    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
//    int tilePos = 0;
//    
//    uint32_t pValueFloat = 0;
//    HAMC_DATA_TYPE_t shortValue = 0;
//    
//    for(int i = 0; i < ((colA - 1)/(TILE_WIDTH * 4)) + 1; i++){
//        tilePos = i * TILE_WIDTH;
//        if((Row < rowA) && (tilePos + threadIdx.x < colA / 4)){
//            sharedFloatA[tid] = floatA[Row * colA / 4 + tilePos + threadIdx.x];
//        }
//        else{
//            sharedFloatA[tid] = (uint32_t)0;
//        }
//        
//        for(int j = 0; j < 4; j++){
//            if(((j * TILE_WIDTH + threadIdx.y + tilePos * 4) < rowB) && (Col < colB)){
//                sharedB[threadIdx.x * 4 * TILE_WIDTH + j * TILE_WIDTH + threadIdx.y] = B[colB * (j * TILE_WIDTH + threadIdx.y + tilePos * 4) + Col];
//            }
//            else{
//                sharedB[threadIdx.x * 4 * TILE_WIDTH + j * TILE_WIDTH + threadIdx.y] = (uint8_t)0;
//            }
//        }
//        __syncthreads();
//        
//        //if(blockIdx.x == 0 && blockIdx.y == 0 && tid == 0){// && i == 0){
//            /*printf("A: i = %d\n", i);
//            
//            for(int q = 0; q < 32; q++){
//                for(int jk = 0; jk < 4; jk++){
//                    for(int k = 0; k < 32; k++){
//                        char bit = (sharedA[q * 4 * TILE_WIDTH + tid + k + jk * TILE_WIDTH]) & 1;
//                        //char bit = (sharedA[q * 4 * TILE_WIDTH + tid + k]) & 1;
//                        printf("%u,", bit);
//                    }
//                }
//                printf("\n");
//            }/**/
//            /*printf("B: i = %d\n", i);
//            for(int q = 0; q < 32; q++){
//                for(int jk = 0; jk < 4; jk++){
//                    for(int k = 0; k < 32; k++){
//                        char bit = (sharedB[q * 4 * TILE_WIDTH + tid + k + jk * TILE_WIDTH]) & 1;
//                        printf("%u,", bit);
//                    }
//                }
//                printf("\n");
//            }/**/
//            
//            /*printf("transposeB 0 through 3: ");
//            for(int k = 0; k < 4; k++){
//                for(int j = 0; j < 8; j++){
//                    char bit = (transposeB[tid + k] >> (7 - j)) & 1;
//                    printf("%u", bit);
//                }
//                printf(" ");
//            }
//            printf("\n");/**/
//        //}
//        for(int j = 0; j < TILE_WIDTH; j++){
//            pValueFloat ^= sharedFloatA[threadIdx.y * TILE_WIDTH + j] & sharedFloatB[threadIdx.x * TILE_WIDTH + j];
//        }/**/
//        __syncthreads();
//    }
//    /*if(blockIdx.x == 0 && blockIdx.y == 0 && tid == 0){
//        printf("pValueFloat: ");
//            for(int j = 0; j < 32; j++){
//                char bit = (pValueFloat >> (31 - j)) & 1;
//                printf("%u", bit);
//            }
//            printf("\n");
//    }/**/
//    if(Row < rowA && Col < colB){
//        for(int i = 0; i < 4; i++){
//            //pValue[0] ^= pValue[i];
//            //shortValue ^= pValue[i] & 1;
//            shortValue ^= pValueFloat & 1;
//            pValueFloat >>= 8;
//        }
//        C[Row * colB + Col] = shortValue;
//    }/**/
//}

__global__ void mult_kernel_compressed_data(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB)
{
    __shared__ uint32_t sharedFloatA[TILE_WIDTH * TILE_WIDTH];
    //__shared__ uint32_t sharedFloatB[TILE_WIDTH * TILE_WIDTH];
    __shared__ uint32_t sharedFloatB[TILE_WIDTH * (TILE_WIDTH + 1)];
    HAMC_DATA_TYPE_t *sharedA = (uint8_t *)sharedFloatA;
    HAMC_DATA_TYPE_t *sharedB = (uint8_t *)sharedFloatB;
    
    uint32_t *floatA = (uint32_t *)A;
    uint32_t *floatB = (uint32_t *)B;
    
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int tilePos = 0;
    
    uint32_t pValueFloat = 0;
    HAMC_DATA_TYPE_t shortValue = 0;
    
    for(int i = 0; i < ((colA - 1)/(TILE_WIDTH * 4)) + 1; i++){
        tilePos = i * TILE_WIDTH;
        /*if((Row < rowA) && (tilePos + threadIdx.x < colA / 4)){
            sharedFloatA[tid] = floatA[Row * colA / 4 + tilePos + threadIdx.x];
        }
        else{
            sharedFloatA[tid] = (uint32_t)0;
        }/**/
        
        /*if((Row < rowA) && (tilePos + threadIdx.x + 3 < colA / 4)){
            //sharedFloatA[tid] = (uint32_t)A[Row * colA + tilePos * 4 + threadIdx.x * 4];
            //sharedFloatA[tid] = *(uint32_t *)((uint8_t *)&floatA[Row * colA / 4 + tilePos + threadIdx.x] + 3 * Row);
            //sharedFloatA[tid] = floatA[Row * colA / 4 + tilePos + threadIdx.x];
            //memcpy(&sharedFloatA[tid], &floatA[Row * (((colA - 1) / 4) + 1) + tilePos + threadIdx.x] + 3 * Row, sizeof(uint32_t));
            //memcpy(&sharedFloatA[tid], &A[Row * colA + tilePos * 4 + threadIdx.x * 4] + 3 * Row, sizeof(uint32_t));
            memcpy(&sharedFloatA[tid], &A[Row * colA + tilePos * 4 + threadIdx.x * 4], sizeof(uint32_t));
        }
        else if((Row < rowA) && (tilePos + threadIdx.x > colA / 4)){
            //sharedFloatA[tid] = floatA[Row * colA / 4 + tilePos + threadIdx.x];
            memcpy(&sharedFloatA[tid], &A[Row * colA + tilePos * 4 + threadIdx.x * 4], sizeof(uint32_t));
            int padding = 4 - colA % 4;
            //printf("Padding: %d\n", padding);
            for(int j = 1; j <= padding; j++){
                sharedA[tid + j] = (uint8_t)0;
            }
        }/**/
        if((Row < rowA) && (tilePos + threadIdx.x < (colA - 1) / 4 + 1)){
            //sharedFloatA[tid] = (uint32_t)A[Row * colA + tilePos * 4 + threadIdx.x * 4];
            //sharedFloatA[tid] = *(uint32_t *)((uint8_t *)&floatA[Row * colA / 4 + tilePos + threadIdx.x] + 3 * Row);
            //sharedFloatA[tid] = floatA[Row * colA / 4 + tilePos + threadIdx.x];
            //memcpy(&sharedFloatA[tid], &floatA[Row * (((colA - 1) / 4) + 1) + tilePos + threadIdx.x] + 3 * Row, sizeof(uint32_t));
            //memcpy(&sharedFloatA[tid], &A[Row * colA + tilePos * 4 + threadIdx.x * 4] + 3 * Row, sizeof(uint32_t));
            memcpy(&sharedFloatA[tid], &A[Row * colA + tilePos * 4 + threadIdx.x * 4], sizeof(uint32_t));
            if((tilePos + threadIdx.x + 1) > colA / 4){
                int padding = colA % 4;
                //if(blockIdx.x == 0 && blockIdx.y == 0) printf("Padding: %d\n", padding);
                for(int j = 3; j >= padding; j--){
                    sharedA[Row * colA + (tilePos + threadIdx.x) * 4 + j] = (uint8_t)0;
                }
            }
        }/**/
        else{
            sharedFloatA[tid] = (uint32_t)0;
        }/**/
        /*for(int j = 0; j < 4; j++){
            #pragma unroll
            if(((j * TILE_WIDTH + threadIdx.y + tilePos * 4) < rowB) && (Col < colB)){
                sharedB[threadIdx.x * 4 * (TILE_WIDTH + 1) + j * (TILE_WIDTH) + threadIdx.y] = B[colB * (j * TILE_WIDTH + threadIdx.y + tilePos * 4) + Col];
                //sharedB[(threadIdx.x * 4 + j) * TILE_WIDTH + threadIdx.y] = B[colB * (j * TILE_WIDTH + threadIdx.y + tilePos * 4) + Col];
            }
            else{
                sharedB[threadIdx.x * 4 * (TILE_WIDTH + 1) + j * (TILE_WIDTH) + threadIdx.y] = (uint8_t)0;
                //sharedB[(threadIdx.x * 4 + j) * TILE_WIDTH + threadIdx.y] = (uint8_t)0;
            }
        }/**/
        
        /*for(int j = 0; j < 4; j++){
            #pragma unroll
            if(((j * TILE_WIDTH + threadIdx.y + tilePos * 4) < rowB) && (Col < colB)){
                sharedB[threadIdx.x * 4 * (TILE_WIDTH + 1) + j * (TILE_WIDTH) + threadIdx.y] = B[colB * (j * TILE_WIDTH + threadIdx.y + tilePos * 4) + Col];
                //sharedB[(threadIdx.x * 4 + j) * TILE_WIDTH + threadIdx.y] = B[colB * (j * TILE_WIDTH + threadIdx.y + tilePos * 4) + Col];
            }
            else{
                sharedB[threadIdx.x * 4 * (TILE_WIDTH + 1) + j * (TILE_WIDTH) + threadIdx.y] = (uint8_t)0;
                //sharedB[(threadIdx.x * 4 + j) * TILE_WIDTH + threadIdx.y] = (uint8_t)0;
            }
        }*/
        
        __syncthreads();
        
        if(blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && i == 0/**/){
            printf("A: i = %d\n", i);
            for(int q = 0; q < 32; q++){
                for(int jk = 0; jk < 4; jk++){
                    for(int k = 0; k < 32; k++){
                        char bit = (sharedA[q * 4 * TILE_WIDTH + tid + k + jk * TILE_WIDTH]) & 1;
                        //char bit = (sharedA[q * 4 * TILE_WIDTH + tid + k]) & 1;
                        printf("%u,", bit);
                    }
                }
                printf("\n");
            }/**/
            
            /*for(int q = 0; q < 32; q++){
                    for(int k = 0; k < 32; k++){
                        for(int asd = 0; asd < 4; asd++){
                            char bit = (sharedFloatA[q * TILE_WIDTH + tid + k] >> (24 - 8 * asd)) & 1;
                            printf("%u,", bit);
                        }
                    }
                printf("\n");
            }/**/
            
            /*for(int q = 0; q < 32; q++){
                for(int k = 0; k < 32; k++){
                    //for(int asd = 0; asd < 4; asd++){
                        //char bit = (sharedFloatA[q * TILE_WIDTH + tid + k] >> (25 - 8 * asd)) & 1;
                        printf("%d,", sharedFloatA[q * TILE_WIDTH + tid + k]);
                    //}
                }
                printf("\n");
            }/**/
            /*printf("B: i = %d\n", i);
            for(int q = 0; q < 32; q++){
                for(int jk = 0; jk < 4; jk++){
                    for(int k = 0; k < 32; k++){
                        char bit = sharedB[(q * 4 + jk) * (TILE_WIDTH + 4) + tid + k] & 1;
                        printf("%u,", bit);
                    }
                }
                printf("\n");
            }/**/
            /*for(int q = 0; q < 32; q++){
                    for(int k = 0; k < 32; k++){
                        for(int asd = 0; asd < 4; asd++){
                            char bit = (sharedFloatB[q * (TILE_WIDTH + 1) + tid + k] >> (31 - 8 * asd)) & 1;
                            printf("%u,", bit);
                        }
                    }
                printf("\n");
            }/**/
        }
        /*for(int j = 0; j < TILE_WIDTH; j++){
            #pragma unroll
            pValueFloat ^= sharedFloatA[threadIdx.y * TILE_WIDTH + j] & sharedFloatB[threadIdx.x * (TILE_WIDTH + 1) + j];
        }/**/
        __syncthreads();
    }
    /*if(blockIdx.x == 0 && blockIdx.y == 0 && tid == 0){
        printf("pValueFloat: ");
            for(int j = 0; j < 32; j++){
                char bit = (pValueFloat >> (31 - j)) & 1;
                printf("%u", bit);
            }
            printf("\n");
    }/**/
    /*if(Row < rowA && Col < colB){
        for(int i = 0; i < 4; i++){
            #pragma unroll
            //pValue[0] ^= pValue[i];
            //shortValue ^= pValue[i] & 1;
            shortValue ^= pValueFloat & 1;
            pValueFloat >>= 8;
        }
        C[Row * colB + Col] = shortValue;
    }/**/
}

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

__global__ void mult_kernel(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB)
{   
    __shared__ HAMC_DATA_TYPE_t sharedA[TILE_WIDTH * TILE_WIDTH];
    __shared__ HAMC_DATA_TYPE_t sharedB[TILE_WIDTH * TILE_WIDTH];
    
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
    if((Row < rowA) && (Col < colB)){
        C[Row * colB + Col] = pValue;
    }
}

__global__ void mult_kernel_debug(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int rowA, int rowB, int colA, int colB)
{
    __shared__ HAMC_DATA_TYPE_t sharedA[TILE_WIDTH * TILE_WIDTH];
    __shared__ HAMC_DATA_TYPE_t sharedB[TILE_WIDTH * TILE_WIDTH];
    
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
        
        if(blockIdx.x == 0 && blockIdx.y == 0 && tid == 0){// && i == 0){
            printf("A: i = %d\n", i);
            for(int q = 0; q < 32; q++){
                for(int k = 0; k < 32; k++){
                    char bit = (sharedA[q * TILE_WIDTH + tid + k]) & 1;
                    printf("%u,", bit);
                }
                printf("\n");
            }/**/
            printf("B: i = %d\n", i);
            for(int q = 0; q < 32; q++){
                for(int k = 0; k < 32; k++){
                    char bit = (sharedB[k * TILE_WIDTH + tid + q]) & 1;
                    printf("%u,", bit);
                }
                printf("\n");
            }/**/
        }
        
        if((Row < rowA) && (Col < colB)){
            for(int j = 0; j < TILE_WIDTH; j++){
                pValue ^= (sharedA[threadIdx.y * TILE_WIDTH + j] & sharedB[j * TILE_WIDTH + threadIdx.x]);
            }
        }
        
        __syncthreads();
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
    //int TILE_WIDTH = 32;
    
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
    
    mult_kernel<<<DimGrid, DimBlock, 2 * TILE_WIDTH * TILE_WIDTH * sizeof(HAMC_DATA_TYPE_t)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols);
    
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

bin_matrix run_mult_kernel(bin_matrix A, bin_matrix B, int hmm)
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
    
    mult_kernel<<<DimGrid, DimBlock, 2 * TILE_WIDTH * TILE_WIDTH * sizeof(HAMC_DATA_TYPE_t)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols);
    
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    cudaMemcpy(C->data, deviceC, C->cols * C->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}

bin_matrix run_mult_kernel_debug(bin_matrix A, bin_matrix B)
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
    
    mult_kernel_debug<<<DimGrid, DimBlock, 2 * TILE_WIDTH * TILE_WIDTH * sizeof(HAMC_DATA_TYPE_t)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols);
    
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    cudaMemcpy(C->data, deviceC, C->cols * C->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}

bin_matrix run_mult_kernel_test(bin_matrix A, bin_matrix B)
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
    
    mult_kernel_compressed_data<<<DimGrid, DimBlock, (2 * TILE_WIDTH * TILE_WIDTH + TILE_WIDTH) * sizeof(float)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols);
    //mult_kernel_compressed_data<<<DimGrid, DimBlock, 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols, TILE_WIDTH);
    //mult_kernel_outer_product<<<DimGrid, DimBlock, TILE_WIDTH * TILE_WIDTH * sizeof(HAMC_DATA_TYPE_t)>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols, TILE_WIDTH);
    
    cudaDeviceSynchronize();
    /*cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    */
    cudaMemcpy(C->data, deviceC, C->cols * C->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}

#endif /* HAMC_MULTIPLY_MATRIX_H */
