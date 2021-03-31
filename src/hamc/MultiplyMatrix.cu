//#include <stdio.h>
#include <iostream>

#define TILE_WIDTH 16
#define ushort int//unsigned short

__global__ void mult_kernel(ushort *A, ushort *B, ushort *C, int rowA, int rowB, int colA, int colB)
{
    __shared__ ushort sharedA[TILE_WIDTH * TILE_WIDTH];
    __shared__ ushort sharedB[TILE_WIDTH * TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int tilePos = 0;

    ushort pValue = 0;
  
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
                //pValue = (pValue ^ (sharedA[threadIdx.y * TILE_WIDTH + j] & sharedB[j * TILE_WIDTH + threadIdx.x]));
                if(sharedA[threadIdx.y * TILE_WIDTH + j] == 0 || sharedB[j * TILE_WIDTH + threadIdx.x] == 0);
                else{
                    if(pValue == 1){
                        pValue = 0;
                    }
                    else{
                        pValue = 1;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    std::cout << "FUCK" << std:endl;
    if((Row < rowA) && (Col < colB)){
        //printf("pValue: %d", pValue);
        C[Row * colB + Col] = pValue;
    }
    
    /*__shared__ ushort sharedA[TILE_WIDTH * TILE_WIDTH];
    __shared__ ushort sharedB[TILE_WIDTH * TILE_WIDTH];
    
    int tilePos = 0;

    ushort pValue = 0;
  
    for(int i = 0; ; i++){
        tilePos = i * TILE_WIDTH;
        if(() && ()){
            sharedA[threadIdx.x] = A[];
        }
        else{
            sharedA[threadIdx.x] = 0.0;
        }
        if(() && ()){
            sharedB[threadIdx.x] = B[];
        }
        else{
            sharedB[threadIdx.x] = 0.0;
        }
        __syncthreads();
        
        if(() && ()){
            for(int j = 0; j < TILE_WIDTH; j++){
                pValue ^= sharedA[] & sharedB[];
            }
        }
        
        __syncthreads();
    }
    if(( < rowA) && ( < colB)){
        C[] = pValue;
    }*/
}