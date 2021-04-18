#include <cuda_runtime.h>
#include<stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>


#include "../../hamc/hamc_common.h"
#include "../../hamc/hamc_cpu_code.c"

#include "MatrixAdd_cpu.c"
#include "../../hamc/MatrixAdd.cu"

#define TILE_WIDTH 16

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

void printHelp()
{
    printf("run this executable with the following flags\n");
    printf("\n");
    printf("\t-i <input file name>\n");
    printf("\t-o <output file name>\n");
    printf("\t-s <solution file name>\n");
}


bin_matrix run_cpu(const char *in, HAMC_DATA_TYPE_t *sol)
{

    int numARows, numAColumns;


    HAMC_DATA_TYPE_t *hostA =
        (HAMC_DATA_TYPE_t *)wbImport(in, &numARows, &numAColumns);

    HAMC_DATA_TYPE_t *hostB =
        (HAMC_DATA_TYPE_t *)malloc(numARows*numAColumns * sizeof(HAMC_DATA_TYPE_t));

    bin_matrix A = mat_init_cpu(numARows, numAColumns);
    A->data = hostA;

    bin_matrix B = mat_init_cpu(numARows, numAColumns);
    B->data = hostB;


    //matrix_add(hostA, hostC, numARows, numAColumns);
    bin_matrix C = add_matrix_cpu(A, B);
    return C;
}


int main(int argc, char *argv[])
{
    printf("MatrixAdd test:\n");
    wbArg_t args;

    HAMC_DATA_TYPE_t *hostA; // The A matrix
    HAMC_DATA_TYPE_t *hostC; // The output C matrix
    HAMC_DATA_TYPE_t *deviceA; // A matrix on device
    HAMC_DATA_TYPE_t *deviceB; // B matrix on device (copy of A)
    HAMC_DATA_TYPE_t *deviceC; // C matrix on device
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A

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
    hostA = (HAMC_DATA_TYPE_t *)wbImport(inputFileName, &numARows, &numAColumns);
    int numBRows = numARows;    // number of rows in the matrix B
    int numBColumns = numAColumns; // number of columns in the matrix B
    int numCRows = numARows;    // number of rows in the matrix C
    int numCColumns = numAColumns; // number of columns in the matrix C
    hostC = (HAMC_DATA_TYPE_t *)malloc(numCRows*numCColumns * sizeof(HAMC_DATA_TYPE_t));
    wbTime_stop(Generic, "Importing data and creating memory on host");


    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);


    /* allocate the memory space on GPU */
    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void**) &deviceB, numBRows * numBColumns * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(HAMC_DATA_TYPE_t));
    wbTime_stop(GPU, "Allocating GPU memory.");


    dim3 dimGrid((numCColumns - 1) / 16 + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    /* call CUDA kernel to perform computations */
    wbTime_start(Compute, "Performing CUDA computation for RREF");
    MatrixAdd_kernel<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");



    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    /* Free GPU Memory */
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceA);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

