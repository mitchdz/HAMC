#include <cuda_runtime.h>
#include <stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include "../../hamc/hamc_cpu_code.c"
#include "../../hamc/TransposeMatrix.cu"

#define TILE_WIDTH 16
#define ushort unsigned short

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess){
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

void printHelp()
{
    printf("run this executable with the following flags\n");
    printf("\n");
    printf("\t-a <input0 file name>\n");
    printf("\t-b <input1 file name>\n");
    printf("\t-e <expected solution file name>\n");
    printf("\t-o <output file name>\n");
    printf("\t-c \n");
}


bin_matrix run_cpu(bin_matrix A)
{
    return transpose_cpu(A);
}

bin_matrix run_kernel(bin_matrix A)
{
    /* ushort *deviceA;
    ushort *deviceB;
    ushort *deviceC; */
    ushort *deviceA;
    ushort *deviceB;
    bin_matrix B = mat_init_cpu(A->rows, A->cols);
    ushort *tempA = (ushort *)malloc(sizeof(ushort) * A->rows * A->cols);
    ushort *tempB = (ushort *)malloc(sizeof(ushort) * B->rows * B->cols);

    for(int i = 0; i < A->rows * A->cols; i++){
        tempA[i] = (ushort)A->data[i];
    }
    for(int i = 0; i < B->rows * B->cols; i++){
        tempB[i] = (ushort)B->data[i];
    }

    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(ushort));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(ushort));

    /* cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(ushort), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(ushort), cudaMemcpyHostToDevice); */
    cudaMemcpy(deviceA, tempA, A->cols * A->rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, tempB, B->cols * B->rows * sizeof(int), cudaMemcpyHostToDevice);

    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((B->cols - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);

    transpose<<<DimGrid, DimBlock>>>(deviceA, deviceB, A->rows, A->cols);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", 
                cudaGetErrorString(cudaerr));
    }

    cudaMemcpy(tempB, deviceB, B->cols * B->rows * sizeof(int),
            cudaMemcpyDeviceToHost);

    for(int i = 0; i < B->rows * B->cols; i++){
        B->data[i] = (ushort)tempB[i];
    }

    /* std::cout << "C->data";
    for(int i = 0; i < (C->rows * C->cols); i++){
        if(i % TILE_WIDTH == 0) std::cout << std::endl;
        std::cout << tempC[i] << " ";
    }
    std::cout << std::endl; */

    cudaFree(deviceA);
    cudaFree(deviceB);

    return B;
}

int main(int argc, char *argv[])
{
    wbArg_t args;
    bin_matrix A;
    bin_matrix B;
    bin_matrix C;
    int numRowsA;
    int numColsA;
    int numRowsB;
    int numColsB;
    int numRowsS;
    int numColsS;
    ushort *hostA;
    ushort *hostB;
    ushort *sol;
    char *input0;
    char *expected;
    bool cpu_exec = false;
    bool solved = true;

    int opt;
    while ((opt = getopt(argc, argv, "i:b:e:o:c")) != -1){
        switch(opt){
            case 'i':
                input0 = strdup(optarg);
                break;
            case 'e':
                expected = strdup(optarg);
                break;
            case 'o':
                //input0 = strdup(optarg);
                break;
            case 'c':
                cpu_exec = true;
                break;
            case 'h':
            default:
                printHelp();
                return 0;
        }
    }
    float *floatTemp = (float *)wbImport(input0, &numRowsA, &numColsA);
    hostA = (ushort *)malloc(numRowsA*numColsA * sizeof(ushort));
    for(int i = 0; i < numColsA * numRowsA; i++){
        hostA[i] = (ushort)floatTemp[i];
    }
    A = mat_init_cpu(numRowsA, numColsA);
    A->data = hostA;

    floatTemp = (float *)wbImport(expected, &numRowsS, &numColsS);
    sol = (ushort *)malloc(numRowsS*numColsS * sizeof(ushort));
    for(int i = 0; i < numColsB * numRowsB; i++){
        sol[i] = (ushort)floatTemp[i];
    }

    /* std::cout << "A->data";
    for(int i = 0; i < numColsA * numRowsA; i++){
        if(i%16 == 0) std::cout << "" << std::endl;
        std::cout << hostA[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "B->data";
    for(int i = 0; i < numColsB * numRowsB; i++){
        if(i%16 == 0) std::cout << "" << std::endl;
        std::cout << hostB[i] << " ";
    }
    std::cout << std::endl; */

    if(cpu_exec) C = run_cpu(A);
    else C = run_kernel(A);

    std::cout << "C->data";
    for(int i = 0; i < C->cols * C->rows; i++){
        if(i%16 == 0) std::cout << "" << std::endl;
        std::cout << C->data[i] << " ";
    }
    std::cout << std::endl;

    if(C->rows != numRowsS && C->cols != numColsS){
        solved = false;
    }
    else{
        for(int i = 0; i < numRowsS * numColsS; i++){
            if(C->data[i] != sol[i]){
                std::cout << "i: " << i << std::endl;
                std::cout << "C->data[i]: " << C->data[i] << std::endl;
                std::cout << "expected: " << sol[i] << std::endl;
                solved = false;
                break;
            }
        }
    }

    /*std::cout << "C->rows: " << C->rows << std::endl;
    std::cout << "C->cols: " << C->cols << std::endl;
    std::cout << "numRowsS: " << numRowsS << std::endl;
    std::cout << "numColsS: " << numColsS << std::endl;*/
    std::cout << "solved: " << solved << std::endl;

    free(A);
    free(B);
    free(C);

    return 0;
}
