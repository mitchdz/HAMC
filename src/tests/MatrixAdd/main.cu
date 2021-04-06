#include <cuda_runtime.h>
#include <stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <time.h>

#include "../../hamc/hamc_cpu_code.c"
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
       
    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);
    
    // Initialize a C Matrix
    // Not Here?
    
    // Run CPU Operation
    cudaEventRecord(astartEvent, 0);
    bin_matrix hostC = matrix_add_cpu(A, B);
    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    
    printf("\n");
    printf("Total compute time (ms) %f for Matrix Add CPU\n",aelapsedTime);
    printf("\n");
    
    return hostC;
}

bin_matrix run_kernel(bin_matrix A, bin_matrix B)
{
    if (A->rows != B->rows || A->cols != B->cols){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }

    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);
    

    
    ushort *deviceA;
    ushort *deviceB;
    ushort *deviceC;
    

    
    bin_matrix C = mat_init_cpu(A->rows,B->cols);

    //C->rows = A->rows;

    //C->cols = B->cols;

    
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
    
    cudaEventRecord(astartEvent, 0);
    matrixAdd_kernel<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, A->rows, A->cols);
    cudaDeviceSynchronize();
    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    
    printf("\n");
    printf("Total compute time (ms) %f for Matrix Add GPU\n",aelapsedTime);
    printf("\n");
    
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

// main - should only be handling the initial matrices A and B generation and input files
int main(int argc, char *argv[])
{
    

    // Variables - Rows & Cols
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    
 
    // Inputs
    wbArg_t args;
    
    // Output
    bin_matrix hostC;
    
 
    char *inputFileName0;
    char *inputFileName1;
    char *solutionFileName;
    

    bool cpu_run = false;

    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;
    
    // Give Input Files
   
    int c;
    opterr = 0;
    while ((c = getopt (argc, argv, "i:j:s:hp")) != -1){
        switch(c)
        {
            case 'i':
                inputFileName0 = strdup(optarg);
                break;
            case 'j':
                inputFileName1 = strdup(optarg);
                break;
            case 's':
                solutionFileName = strdup(optarg);
                break;
            case 'h':
                printHelp();
                return 0;
            case 'n':
            case 'p':
                cpu_run = true;
                break;
            default:
                abort();
        }
   }
    args = wbArg_read(argc, argv);
    
    // Read Input Files
        // Input File 0 - Matrix A
    
    float *hostAFloats = (float *)wbImport(inputFileName0, &numARows, &numAColumns);
    ushort *hostA = (ushort *)malloc(numARows*numAColumns * sizeof(ushort));
    for (int i = 0; i < numARows*numAColumns; i++)
    {
        hostA[i] = (ushort)hostAFloats[i];
    }
        
        // Input File 1 - Matrix B
    float *hostBFloats = (float *)wbImport(inputFileName1, &numBRows, &numBColumns);
    
    ushort *hostB = (ushort *)malloc(numBRows*numBColumns * sizeof(ushort));
   
    for (int i = 0; i < numBRows*numBColumns; i++)
    {
        hostB[i] = (ushort)hostBFloats[i];
    }
    
    // Input File 1 - Solution File
    float *hostOutputFile = (float *)wbImport(solutionFileName, &numBRows, &numBColumns);
    
    ushort *hostOutput = (ushort *)malloc(numBRows*numBColumns * sizeof(ushort));
   
    for (int i = 0; i < numBRows*numBColumns; i++)
    {
        hostOutput[i] = (ushort)hostOutputFile[i];
    }
    
    
    printf("8 \n\n\n");  
    // Initilizing the Matricies HERE!
    
    // Matrix A
    
    bin_matrix hostABin = mat_init_cpu(numARows, numAColumns);
    
    for (int i = 0; i < numARows*numAColumns; i++) 
    {
        hostABin->data[i] = hostA[i];
    }
    
    // Matrix B
  
    bin_matrix hostBBin = mat_init_cpu(numBRows, numBColumns);
    
    for (int i = 0; i < numBRows*numBColumns; i++) 
    {
       hostBBin->data[i] = hostB[i];
    }  
    
    // Call Either Kernels
    if(cpu_run)
    {
        hostC = run_cpu(hostABin, hostBBin);
        
    } else {

        hostC = run_kernel(hostABin,hostBBin);
        
    }
        
    // Check Solution
    
   
    //wbSolution(args, hostC, numARows, numAColumns);
   
    for(int i = 0; i < numARows*numAColumns; i++)
    {
    	if(hostC->data[i] != hostOutput[i])
    	{
    	   printf("Index: %i \n", i);
    	   printf("Kernel Output: %i \n", hostC->data[i]);
    	   printf("Expected: %i \n", hostOutput[i]);
    	}
    } 
    
    free(hostABin);
    free(hostBBin);
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
