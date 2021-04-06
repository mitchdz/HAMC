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
#include "../../hamc/TransposeMatrix.cu"

#define TILE_WIDTH 16
#define BLOCK_DIM 16
#define BLOCK_SIZE 16
#define ushort unsigned short

#define TILE_DIM 16
#define BLOCK_ROWS 8


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
    printf("\t-e <expected solution file name>\n");
    printf("\t-c \n");
    printf("\t  run CPU based execution\n");
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

void run_test(int x, int y)
{

    clock_t start, end;
    double cpu_time_used;

    ushort *raw_data = (ushort *)malloc(sizeof(ushort) * x * y);
    raw_data = generate_data(x, y);

    bin_matrix input = mat_init_cpu(x,y);
    input->data = raw_data;

    /* CPU execution time */
    start = clock();

    bin_matrix CPU_BIN = transpose_cpu(input);

    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("CPU time: %lf\n", cpu_time_used);


    /* GPU execution time */
    start = clock();

    bin_matrix GPU_BIN = run_transpose_kernel(input);

    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("GPU time: %lf\n", cpu_time_used);

    free(input);
    free(raw_data);
    free(CPU_BIN);
    free(GPU_BIN);
}



int main(int argc, char *argv[])
{
    printf("Transpose matrix unit test\n");

    bin_matrix A;
    bin_matrix B;
    int numRowsA;
    int numColsA;
    int numRowsS;
    int numColsS;
    ushort *hostA;
    ushort *sol;
    char *input0 = NULL;
    char *expected = NULL;
    bool cpu_exec = false;
    bool solved = true;

    char *action = NULL;
    int x, y;

    int opt;
    while ((opt = getopt(argc, argv, "i:b:e:o:cx:y:a:")) != -1){
        switch(opt){
            case 'y':
                y = atoi(optarg);
                break;
             case 'x':
                x = atoi(optarg);
                break;
            case 'a':
                action = strdup(optarg);
                break;
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

    if (!strcmp(action, (const char*)"test")) {
        run_test(x, y);
        return 0;
    }

    if (!input0|| !expected) {
        printf("Invalid inputs.\n");
        return -1;
    }


    printf("input file: %s\n", input0);
    printf("solution fil: %s\n", expected);

    printf("Reading input file...\n");
    float *floatTemp = (float *)wbImport(input0, &numRowsA, &numColsA);
    hostA = (ushort *)malloc(numRowsA*numColsA * sizeof(ushort));
    for(int i = 0; i < numColsA * numRowsA; i++){
        hostA[i] = (ushort)floatTemp[i];
    }
    A = mat_init_cpu(numRowsA, numColsA);
    A->data = hostA;

    printf("Reading Solution file...\n");
    float *floatTemp2 = (float *)wbImport(expected, &numRowsS, &numColsS);
    sol = (ushort *)malloc(numRowsS*numColsS * sizeof(ushort));
    for(int i = 0; i < numColsS * numRowsS; i++){
        sol[i] = (ushort)floatTemp2[i];
    }


    printf("Input matrix:\n");
    printf("%d x %d\n", numRowsA, numColsA);
    for (int i = 0; i < numRowsA; i++) {
        for (int j = 0; j < numColsA; j++) {
            printf("%hu ",A->data[i*j + j]);
        }
        printf("\n");
    }

    if(cpu_exec) {
        printf("C Based execution:\n");
        B = transpose_cpu(A);
    }
    else {
        printf("GPU Based execution:\n");
        B = run_transpose_kernel(A);
    }


    printf("\n");
    printf("Solution matrix:\n");
    printf("%d x %d\n", numRowsS, numColsS);
    for (int i = 0; i < numRowsS; i++) {
        for (int j = 0; j < numColsS; j++) {
            printf("%hu ",sol[i*j + j]);
        }
        printf("\n");
    }
    printf("\n");


    printf("Output matrix:\n");
    printf("%d x %d\n", numRowsS, numColsS);
    for (int i = 0; i < numRowsS; i++) {
        for (int j = 0; j < numColsS; j++) {
            printf("%hu ",B->data[i*j + j]);
        }
        printf("\n");
    }


    if(B->rows != numRowsS && B->cols != numColsS){
        solved = false;
    }
    else{
        for(int i = 0; i < numRowsS * numColsS; i++){
            if(B->data[i] != sol[i]){
                std::cout << "i: " << i << std::endl;
                std::cout << "C->data[i]: " << B->data[i] << std::endl;
                std::cout << "expected: " << sol[i] << std::endl;
                solved = false;
                break;
            }
        }
    }

    std::cout << "solved: " << solved << std::endl;

    free(A);
    free(B);

    return 0;
}
