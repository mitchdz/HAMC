#include <cuda_runtime.h>
#include<stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>

#include "RREF_cpu.c"
#include "../../hamc/RREFMatrix.cu"

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
    printf("\t  input filename to run the code against\n");

    printf("\t-s <solution file name>\n");
    printf("\t  filename for the solution file to check against\n");

    printf("\t-c\n");
    printf("\t  runs the CPU based execution with timing\n");

    printf("\t-h\n");
    printf("\t  prints this help menu\n");
}


void run_cpu(char *in, char*sol, bool verbose)
{
    cudaEvent_t astartEvent, astopEvent;

    int numARows, numAColumns;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);

    printf("input file: %s\n",in );
    printf("solution file: %s\n",sol );

    /* wbImport only reads and writes float, so we need to convert that */
    float *hostAFloats = (float *)wbImport(in, &numARows, &numAColumns);
    ushort *hostA = (ushort *)malloc(numARows*numAColumns * sizeof(ushort));
    for (int i = 0; i < numARows*numAColumns; i++)
        hostA[i] = (ushort)hostAFloats[i];

    //ushort *hostC = (ushort *)malloc(numARows*numAColumns * sizeof(ushort));

    if (verbose) {
        /* print input array */
        printf("Input Array:\n");
        for( int i = 0; i < numARows; i++) {
            for (int j = 0; j < numAColumns; j++) {
                printf("%u ", hostA[i*numAColumns + j]);
            }
            printf("\n");
        }
    }

    bin_matrix hostABin = mat_init(numARows, numAColumns);
    for (int i = 0; i < numARows*numAColumns; i++) {
        hostABin->data[i] = hostA[i];
    }
    if (verbose) {
        /* print input array */
        printf("Input Array:\n");
        for( int i = 0; i < numARows; i++) {
            for (int j = 0; j < numAColumns; j++) {
                printf("%u ", hostABin->data[i*numAColumns + j]);
            }
            printf("\n");
        }
    }


    cudaEventRecord(astartEvent, 0);
    bin_matrix hostCBin = matrix_rref(hostABin);
    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);

    if (verbose) {
        /* print solution array */
        printf("Solution Array:\n");
        for( int i = 0; i < numARows; i++) {
            for (int j = 0; j < numAColumns; j++) {
                printf("%u ", hostCBin->data[i*numAColumns + j]);
            }
            printf("\n");
        }
    }


    printf("\n");
    printf("Total compute time (ms) %f for RREF cpu\n",aelapsedTime);
    printf("\n");

cleanup:
    free(hostAFloats);
    free(hostABin);
    free(hostCBin);
    free(hostA);

}


int main(int argc, char *argv[])
{
    printf("RREF test:\n");
    wbArg_t args;

    ushort *hostA; // The A matrix
    ushort *hostC; // The output C matrix
    ushort *deviceA; // A matrix on device
    ushort *deviceB; // B matrix on device (copy of A)
    ushort *deviceC; // C matrix on device
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A

    char *inputFileName;
    char *solutionFileName;

    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);

    bool run_cpu_flag = false;

    int c;
    opterr = 0;
    while ((c = getopt (argc, argv, "i:s:hc")) != -1)
        switch(c)
        {
            case 'i':
                inputFileName = strdup((const char*)optarg);
                break;
            case 's':
                solutionFileName = strdup((const char*)optarg);
                break;
            case 'h':
                printHelp();
                return 0;
            case 'c':
                run_cpu_flag = true;
                break;
            default:
                abort();
        }

    args = wbArg_read(argc, argv);

    if (run_cpu_flag){
        run_cpu(inputFileName, solutionFileName, true);
        return 0;
    }


    printf("input file: %s\n", inputFileName);
    printf("solution file: %s\n", solutionFileName);



    /* allocate host data for matrix */
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (ushort *)wbImport(inputFileName, &numARows, &numAColumns);
    int numBRows = numARows;    // number of rows in the matrix B
    int numBColumns = numAColumns; // number of columns in the matrix B
    int numCRows = numARows;    // number of rows in the matrix C
    int numCColumns = numAColumns; // number of columns in the matrix C
    hostC = (ushort *)malloc(numCRows*numCColumns * sizeof(ushort));
    wbTime_stop(Generic, "Importing data and creating memory on host");


    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);


    /* allocate the memory space on GPU */
    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(ushort));
    cudaMalloc((void**) &deviceB, numBRows * numBColumns * sizeof(ushort));
    cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(ushort));
    wbTime_stop(GPU, "Allocating GPU memory.");


    dim3 dimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    /* call CUDA kernel to perform computations */
    wbTime_start(Compute, "Performing CUDA computation for RREF");
    rref_kernel<<<dimGrid, dimBlock>>>(deviceA, deviceB, numARows, numAColumns, deviceC);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");



    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(ushort), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    /* Free GPU Memory */
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostC);

    return 0;
}
