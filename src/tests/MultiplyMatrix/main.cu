#include <cuda_runtime.h>
#include <stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include "../../hamc/hamc_cpu_code.c"
#include "../../hamc/MultiplyMatrix.cu"

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


bin_matrix run_cpu(bin_matrix A, bin_matrix B)
{
    return matrix_mult_cpu(A, B);
}

/*void unit_test(bin_matrix A, bin_matrix B, bool cpu_exec)
{
    if(cpu_exec){
        C = run_cpu(A, B);
    }
    else{
        std::cout << "Running Kernel" << std::endl;
        C = run_kernel(A, B);
    }
    
    if(C->rows != numRowsS && C->cols != numColsS){
        solved = false;
    }
    else{
        for(int i = 0; i < numRowsS * numColsS; i++){
            if(C->data[i] != sol[i]){
                solved = false;
                break;
            }
        }
    }
    
    std::cout << "solved: " << solved << std::endl;
}*/

void time_test(int x, int y)
{
    clock_t start, end;
    double time_used;
    
    ushort *dataA = (ushort *)malloc(sizeof(ushort) * x * y);
    ushort *dataB = (ushort *)malloc(sizeof(ushort) * x * y);
    
    //TODO: add datast gen
    for(int i = 0; i < x * y; i++){
        dataA[i] = (ushort)(rand() % 2);
        dataB[i] = (ushort)(rand() % 2);
    }
    
    bin_matrix A = mat_init_cpu(x, y);
    bin_matrix B = mat_init_cpu(y, x);
    
    A->data = dataA;
    B->data = dataB;
    
    start = clock();
    
    bin_matrix C = run_cpu(A, B);
    
    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    std::cout << "CPU time: " << time_used << std::endl;
    
    free(C);
    
    start = clock();
    
    C = run_mult_kernel(A, B);
    
    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    std::cout << "GPU time: " << time_used << std::endl;
}

int main(int argc, char *argv[])
{
    //wbArg_t args;
    bin_matrix A;
    bin_matrix B;
    bin_matrix C;
    int numRowsA;
    int numColsA;
    int numRowsB;
    int numColsB;
    int numRowsS;
    int numColsS;
    int x, y;
    ushort *hostA;
    ushort *hostB;
    ushort *sol;
    char *input0;
    char *input1;
    char *expected;
    bool cpu_exec = false;
    bool trial_time = false;
    bool solved = true;
    
    int opt;
    while ((opt = getopt(argc, argv, "a:b:e:o:x:y:tc")) != -1){
        switch(opt){
            case 'a':
                input0 = strdup(optarg);
                break;
            case 'b':
                input1 = strdup(optarg);
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
            case 't':
                trial_time = true;
                break;
            case 'x':
                x = atoi(optarg);
                break;
            case 'y':
                y = atoi(optarg);
                break;
            case 'h':
            default:
                printHelp();
                return 0;
        }
    }
    
    if(trial_time){
        time_test(x, y);
        return 0;
    }
    
    float *floatTemp = (float *)wbImport(input0, &numRowsA, &numColsA);
    hostA = (ushort *)malloc(numRowsA*numColsA * sizeof(ushort));
    for(int i = 0; i < numColsA * numRowsA; i++){
        hostA[i] = (ushort)floatTemp[i];
    }
    A = mat_init_cpu(numRowsA, numColsA);
    A->data = hostA;
    
    floatTemp = (float *)wbImport(input1, &numRowsB, &numColsB);
    hostB = (ushort *)malloc(numRowsB*numColsB * sizeof(ushort));
    for(int i = 0; i < numColsB * numRowsB; i++){
        hostB[i] = (ushort)floatTemp[i];
    }    
    B = mat_init_cpu(numRowsB, numColsB);
    B->data = hostB;
    
    floatTemp = (float *)wbImport(expected, &numRowsS, &numColsS);
    sol = (ushort *)malloc(numRowsS*numColsS * sizeof(ushort));
    for(int i = 0; i < numColsB * numRowsB; i++){
        sol[i] = (ushort)floatTemp[i];
    }    
    
    if(cpu_exec){
        C = run_cpu(A, B);
    }
    else{
        std::cout << "Running Kernel" << std::endl;
        C = run_mult_kernel(A, B);
    }
    
    if(C->rows != numRowsS && C->cols != numColsS){
        solved = false;
    }
    else{
        for(int i = 0; i < numRowsS * numColsS; i++){
            if(C->data[i] != sol[i]){
                solved = false;
                break;
            }
        }
    }
    
    std::cout << "solved: " << solved << std::endl;
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}