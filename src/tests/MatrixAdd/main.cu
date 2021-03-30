#include <cuda_runtime.h>
#include<stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>

#include "MatrixAdd_cpu.h"
#include "../../hamc/MatrixAdd.cu"

#define TILE_WIDTH 16
#define ushort unsigned short

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

#define mat_element(mat, row_idx, col_idx) \
    mat->data[row_idx * (mat->cols) + col_idx]

typedef struct matrix
{
    int rows;
    int cols;
    ushort *data;

}*bin_matrix;

void* safe_malloc(size_t n)
{
    void* p = malloc(n);
    if (!p)
    {
        fprintf(stderr, "Out of memory(%lu bytes)\n",(size_t)n);
        exit(EXIT_FAILURE);
    }
    return p;
}

void printHelp()
{
    printf("run this executable with the following flags\n");
    printf("\n");
    printf("\t-i <input file name>\n");
    printf("\t-o <output file name>\n");
    printf("\t-s <solution file name>\n");
}

bin_matrix run_cpu(bin_matrix A, bin_matrix B)
{
    if (A->rows != B->rows || A->cols != B->cols){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }
    
    return MatrixAdd_cpu(A, B);
}

bin_matrix run_kernel(bin_matrix A, bin_matrix B)
{
    if (A->rows != B->rows || A->cols != B->cols){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }
    
    ushort *deviceA;
    ushort *deviceB;
    ushort *deviceC;
    
    /* allocate the memory space on GPU */
    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(ushort));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(ushort));
    cudaMalloc((void **) &deviceC, B->cols * A->rows * sizeof(ushort));
    wbTime_stop(GPU, "Allocating GPU memory.");
    
       
    cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(ushort), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(ushort), cudaMemcpyHostToDevice);
    
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((B->cols - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);
    
    matrixAdd<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, A->rows, A->cols);
    
    cudaDeviceSynchronize();
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(C->data, deviceC, B->cols * A->rows * sizeof(ushort), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");
    
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    return C;
}


int main(int argc, char *argv[])
{
    printf("MatrixAdd test:\n");
    
    // Variable - Matrices (Device)
    wbArg_t args;
    bin_matrix A;
    bin_matrix B;
    bin_matrix C;
    
    // Variable - Matrices (Host)
    ushort *hostA;
    ushort *hostB;
    ushort *hostC;
    
    // Variables - Rows & Cols
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;    // number of rows in the matrix C
    int numCColumns; // number of columns in the matrix C
    
    //Inputs
    char *inputFileName;
    char *solutionFileName;

    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);

    int c;
    opterr = 0;
    printf("1\n");
    while ((c = getopt (argc, argv, "i:s:h")) != -1)
        switch(c)
        {
            case 'i':
                inputFileName = strdup(optarg);
                break;
            case 's':
                solutionFileName = strdup(optarg);
                break;
            case 'h':
                printHelp();
                return 0;
            default:
                abort();
        }

    args = wbArg_read(argc, argv);


    printf("input file: %s\n", inputFileName);
    printf("solution file: %s\n", solutionFileName);


    wbTime_start(Compute, "Performing CPU computation for MatrixAdd");
    run_cpu(inputFileName, solutionFileName);
    wbTime_stop(Compute, "Performing CPU computation");


    /* allocate host data for matrix */
    wbTime_start(Generic, "Importing data and creating memory on host");
    
    hostA = (ushort *)wbImport(inputFileName, &numARows, &numAColumns);
    A = mat_init(numARows, numACols);
    A->data = hostA;
    
    hostB = (ushort *)wbImport(inputFileName, &numBRows, &numBColumns);
    B = mat_init(numARows, numACols);
    B->data = hostB;
    
    hostC = (ushort *)malloc(numCRows*numCColumns * sizeof(ushort));
    
    wbTime_stop(Generic, "Importing data and creating memory on host");


    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

