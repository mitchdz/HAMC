#include "../test.h"
#DEFINE TILE_WIDTH 16

//TODO: uint_16t

__global__ void mult_kernel(ushort *A, ushort *B, ushort *C, int rowA, int rowB, int colA, int colB)
{
    __shared__ ushort sharedA[TILE_WIDTH * TILE_WIDTH];
    __shared__ ushort sharedB[TILE_WIDTH * TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int tilePos = 0;

    float pValue = 0;
  
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
                pValue ^= sharedA[threadIdx.y * TILE_WIDTH + j] & sharedB[j * TILE_WIDTH + threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    if((Row < rowA) && (Col < colB)){
        C[Row * colB + Col] = pValue;
    }
}

bin_matrix matrix_mult(bin_matrix A, bin_matrix B){
    if (A->cols != B->rows){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }

    ushort *deviceA;
    ushort *deviceB;
    ushort *deviceC;
    //bin_matrix C;
    //C = mat_init(A->rows, B->cols);
    /*bin_matrix B_temp = transpose(B);

    for(int i = 0; i < A->rows; i++){
        for(int j = 0  ; j < B->cols; j++){
            unsigned short val = 0;
            for(int k = 0; k < B->rows; k++)
                val = (val ^ (mat_element(A, i, k) & mat_element(B_temp, j, k)));
            mat_element(C, i, j) = val;
        }
    }*/
    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(ushort));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(ushort));
    cudaMalloc((void **) &deviceC, B->cols * A->rows * sizeof(ushort));
    
    cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(ushort), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(ushort), cudaMemcpyHostToDevice);
    
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((B->cols - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);
    
    matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(C->data, deviceC, B->cols * A->rows * sizeof(ushort), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}