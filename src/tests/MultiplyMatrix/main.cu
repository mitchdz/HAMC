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

void run_time(int x, int y)
{
    clock_t start, end;
    double time_used;
    
    HAMC_DATA_TYPE_t *dataA = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t) * x * y);
    HAMC_DATA_TYPE_t *dataB = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t) * x * y);
    
    for(int i = 0; i < x * y; i++){
        dataA[i] = (HAMC_DATA_TYPE_t)(rand() % 2);
        dataB[i] = (HAMC_DATA_TYPE_t)(rand() % 2);
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
    
    C = run_mult_kernel(A, B, 16);
    
    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    std::cout << "GPU time: " << time_used << std::endl;
}

void run_tile_sweep(int x, int y, int upto)
{
    clock_t start, end;
    double time_used;
    
    HAMC_DATA_TYPE_t *dataA = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t) * x * y);
    HAMC_DATA_TYPE_t *dataB = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t) * x * y);
    
    for(int i = 0; i < x * y; i++){
        dataA[i] = (HAMC_DATA_TYPE_t)(rand() % 2);
        dataB[i] = (HAMC_DATA_TYPE_t)(rand() % 2);
    }
    
    bin_matrix A = mat_init_cpu(x, y);
    bin_matrix B = mat_init_cpu(y, x);
    
    A->data = dataA;
    B->data = dataB;
    for(int i = 16; i <= upto; i *= 2){
        start = clock();
    
        C = run_mult_kernel(A, B, 16);
        
        end = clock();
        time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "GPU time: " << time_used << std::endl;
        }
        start = clock();
        
        C = run_mult_kernel(A, B, 16);
        
        end = clock();
        time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "GPU time: " << time_used << std::endl;
}

void run_size_sweep()
{
    
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
    int x, y, upto;
    HAMC_DATA_TYPE_t *hostA;
    HAMC_DATA_TYPE_t *hostB;
    HAMC_DATA_TYPE_t *sol;
    char *input0;
    char *input1;
    char *expected;
    bool cpu_exec = false;
    bool trial_time = false;
    bool sweep_tile_test = false;
    bool sweep_size_test = false;
    bool solved = true;
    
    int opt;
    while ((opt = getopt(argc, argv, "a:b:e:o:ctsdx:y:u:h")) != -1){
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
            case 's':
                sweep_tile_test = true;
                break;
            case 'd':
                sweep_size_test = true;
            case 'x':
                x = atoi(optarg);
                break;
            case 'y':
                y = atoi(optarg);
                break;
            case 'u':
                upto = atoi(optarg);
                break;
            case 'h':
            default:
                printHelp();
                return 0;
        }
    }
    
    if(trial_time){
        run_time(x, y);
        return 0;
    }
    if(sweep_tile_test){
        run_tile_sweep(x, y, upto);
        return 0;
    }
    /*if(sweep_size_test){
        run_size_sweep();
        return 0;
    }*/
    
    float *floatTemp = (float *)wbImport(input0, &numRowsA, &numColsA);
    hostA = (HAMC_DATA_TYPE_t *)malloc(numRowsA*numColsA * sizeof(HAMC_DATA_TYPE_t));
    for(int i = 0; i < numColsA * numRowsA; i++){
        hostA[i] = (HAMC_DATA_TYPE_t)floatTemp[i];
    }
    A = mat_init_cpu(numRowsA, numColsA);
    A->data = hostA;
    
    floatTemp = (float *)wbImport(input1, &numRowsB, &numColsB);
    hostB = (HAMC_DATA_TYPE_t *)malloc(numRowsB*numColsB * sizeof(HAMC_DATA_TYPE_t));
    for(int i = 0; i < numColsB * numRowsB; i++){
        hostB[i] = (HAMC_DATA_TYPE_t)floatTemp[i];
    }    
    B = mat_init_cpu(numRowsB, numColsB);
    B->data = hostB;
    
    floatTemp = (float *)wbImport(expected, &numRowsS, &numColsS);
    sol = (HAMC_DATA_TYPE_t *)malloc(numRowsS*numColsS * sizeof(HAMC_DATA_TYPE_t));
    for(int i = 0; i < numColsB * numRowsB; i++){
        sol[i] = (HAMC_DATA_TYPE_t)floatTemp[i];
    }    
    
    if(cpu_exec){
        C = run_cpu(A, B);
    }
    else{
        std::cout << "Running Kernel" << std::endl;
        C = run_mult_kernel(A, B, 16);
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
