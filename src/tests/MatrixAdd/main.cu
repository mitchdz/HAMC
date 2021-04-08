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
        
    // Run CPU Operation
    bin_matrix hostC = matrix_add_cpu(A, B);

    
    return hostC;
}

static ushort *generate_data(int height, int width)
{
    ushort *data = (ushort *)malloc(sizeof(ushort) * width * height);
    int i;
    for (i = 0; i < width * height; i++) {
        data[i] = (ushort)(rand() % 2); // 0 or 1
    }
    return data;
}


bin_matrix run_kernel(bin_matrix A, bin_matrix B)
{
    if (A->rows != B->rows || A->cols != B->cols){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }

    //cudaEvent_t astartEvent, astopEvent;
    //float aelapsedTime;
    //cudaEventCreate(&astartEvent);
    //cudaEventCreate(&astopEvent);
    

    
    ushort *deviceA;
    ushort *deviceB;
    ushort *deviceC;
    

    
    bin_matrix C = mat_init_cpu(A->rows,B->cols);

    //C->rows = A->rows;

    //C->cols = B->cols;

    
    /* allocate the memory space on GPU */
  //  wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(ushort));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(ushort));
    cudaMalloc((void **) &deviceC, B->cols * A->rows * sizeof(ushort));
   // wbTime_stop(GPU, "Allocating GPU memory.");
    

    
    cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(ushort), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(ushort), cudaMemcpyHostToDevice);
    
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((B->cols - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);
    
   // cudaEventRecord(astartEvent, 0);
    matrixAdd_kernel<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, A->rows, A->cols);
    //cudaDeviceSynchronize();
    //cudaEventRecord(astopEvent, 0);
   // cudaEventSynchronize(astopEvent);
   // cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    
    //printf("\n");
    //printf("Total compute time (ms) %f for Matrix Add GPU\n\n",aelapsedTime);
    //printf("\n");
    
   // wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(C->data, deviceC, B->cols * A->rows * sizeof(ushort), cudaMemcpyDeviceToHost);
    //wbTime_stop(Copy, "Copying output memory to the CPU");
    
    //wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    //wbTime_stop(GPU, "Freeing GPU Memory");

    return C;
}

void run_test(int x, int y)
{

    clock_t start, end;
    double cpu_time_used;
    
    
    printf("X var = %i \n", x);
    printf("Y var = %i \n", y);
    
    
    // Matrix A
    ushort *raw_data0 = (ushort *)malloc(sizeof(ushort) * x * y);
    raw_data0 = generate_data(x, y);

    bin_matrix input0 = mat_init_cpu(x,y);
    input0->data = raw_data0;
    
    //Matrix B
    ushort *raw_data1 = (ushort *)malloc(sizeof(ushort) * x * y);
    raw_data1 = generate_data(x, y);

    bin_matrix input1 = mat_init_cpu(x,y);
    input1->data = raw_data1;

    /* CPU execution time */
    start = clock();

    bin_matrix CPU_BIN = run_cpu(input0, input1);

    end = clock();
    
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("CPU time: %lf \n", cpu_time_used);


    /* GPU execution time */
    start = clock();

    bin_matrix GPU_BIN = run_kernel(input0, input1);

    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("GPU time: %lf \n", cpu_time_used);

    free(input0);
    free(input1);
    free(raw_data0);
    free(raw_data1);
    free(CPU_BIN);
    free(GPU_BIN);
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
    bool just_test = false;
    
    int x, y;

    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;
    
    // Give Input Files
   
    int c;
    opterr = 0;
    int num_ran = 0;
    while ((c = getopt (argc, argv, "t:x:y:i:j:s:hp")) != -1){
        switch(c)
        {
            case 't':
            	
            	just_test = true;
            	break;
            case 'x':
            	
            	x = atoi(optarg);
            	break;
            case 'y':
            	
            	y = atoi(optarg);
            	break;
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
            	printf("p check \n\n");
                cpu_run = true;
                break;
            default:
                abort();
        }
        num_ran++;
   }
    
    
    if (just_test) {
    	printf("Test is running! \n\n");
        run_test(x, y);
        return 0;
    } else {
    	args = wbArg_read(argc, argv);
    }
    
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
