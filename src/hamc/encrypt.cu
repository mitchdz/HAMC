#ifndef ENCRYPT_KERNEL_H
#define ENCRYPT_KERNEL_H

#include <stdio.h>
#include <time.h>


#include "TransposeMatrix.cu"
#include "MultiplyMatrix.cu"
#include "hamc_cpu_code.c"

#define TILE_WIDTH_MULTIPLY 16

#ifndef ushort
#define ushort unsigned short
#endif

bin_matrix run_matrix_multiply_kernel(bin_matrix A, bin_matrix B)
{
    bin_matrix C = mat_init_cpu(A->rows, B->cols);

    /* allocate device memory */
    ushort *deviceA;
    ushort *deviceB;
    ushort *deviceC;
    cudaMalloc((void **) &deviceA, A->rows * A->cols * sizeof(ushort));
    cudaMalloc((void **) &deviceB, B->rows * B->cols * sizeof(ushort));
    cudaMalloc((void **) &deviceC, C->rows * C->cols * sizeof(ushort));

    /* transfer host data to device */
    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(ushort), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(ushort), cudaMemcpyHostToDevice);

    printf("Starting multiply matrix kernel...\n");

     /* determine block and grid dimensions */
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->cols - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);

    mult_kernel<<<DimGrid, DimBlock>>> (deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(C->data, deviceC, C->rows * C->cols * sizeof(ushort), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}



void run_encryption_gpu(const char* inputFileName, const char* outputFileName,
        int n, int p, int t, int w, int seed)
{
    //TODO:
}

#endif /* ENCRYPT_KERNEL_H */
