
#ifndef TEST_INVERSE_CPU_H
#define TEST_INVERSE_CPU_H


#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <time.h>

#include "../hamc/hamc_cpu_code.c"
#include "../hamc/InverseMatrix.cu"
#include "debug_inverse_c.c"


bin_matrix my_circ_matrix_inverse_cpu(bin_matrix A);

int main(int argc, char *argv[])
{
    printf("Inverse test\n");

    char *input = NULL;
    char *expected = NULL;

    int opt;
    while ((opt = getopt(argc, argv, "i:e:")) != -1){
        switch(opt){
            case 'i':
                input = strdup(optarg);
                break;
            case 'e':
                expected = strdup(optarg);
                break;
        }
    }

    if (!input || !expected) {
        printf("Please supply input file and expected file!\n");
        return -1;
    }


    bin_matrix A;
    HAMC_DATA_TYPE_t *hostA;
    int numRowsA;
    int numColsA;

    printf("Reading input file...\n");
    float *floatTemp = (float *)wbImport(input, &numRowsA, &numColsA);
    hostA = (HAMC_DATA_TYPE_t *)malloc(numRowsA*numColsA * sizeof(HAMC_DATA_TYPE_t));
    for(int i = 0; i < numColsA * numRowsA; i++){
        hostA[i] = (HAMC_DATA_TYPE_t)floatTemp[i];
    }
    A = mat_init_cpu(numRowsA, numColsA);
    A->data = hostA;
    A->rows = numRowsA;
    A->cols = numColsA;

    //bin_matrix sol;
    HAMC_DATA_TYPE_t *sol;
    int numRowsS;
    int numColsS;

    printf("Reading Solution file...\n");
    float *floatTemp2 = (float *)wbImport(expected, &numRowsS, &numColsS);
    sol = (HAMC_DATA_TYPE_t *)malloc(numRowsS*numColsS * sizeof(HAMC_DATA_TYPE_t));
    for(int i = 0; i < numColsS * numRowsS; i++){
        sol[i] = (HAMC_DATA_TYPE_t)floatTemp2[i];
    }

    bin_matrix E = mat_init_cpu(numRowsS, numColsS);
    E->data = sol;

    clock_t cpu_start, cpu_end;
    double cpu_time_used;

    clock_t gpu_start, gpu_end;
    double gpu_time_used;

    //int rows = 4;
    //int cols = 4;

    //bin_matrix msg_cpu = mat_init_cpu(rows, cols);
    //bin_matrix msg_gpu = mat_init_cpu(rows, cols);
    ////Initializing the message a random message
    //for(int i = 0; i < rows; i++) {
    //    for (int j = 0; j < cols; j++) {
    //        int z = rand() % 2;
    //        set_matrix_element_cpu(msg_cpu, i, j, z);
    //        set_matrix_element_cpu(msg_gpu, i, j, z);
    //    }
    //}

    // print random input data:
    printf("input matrix %dx%d:\n", A->rows, A->cols);
    print_bin_matrix(A);


    cpu_start = clock();

    //bin_matrix inverted_cpu = my_circ_matrix_inverse_cpu(A);

    cpu_end = clock();
    cpu_time_used = ((double) (cpu_end - cpu_start))/ CLOCKS_PER_SEC;

    gpu_start = clock();

    bin_matrix inverted_gpu = run_inverse_kernel(A);
    //bin_matrix inverted_gpu = circ_matrix_inverse_cpu(A);

    gpu_end = clock();
    gpu_time_used = ((double) (gpu_end - gpu_start))/ CLOCKS_PER_SEC;


    printf("Expected matrix:\n");
    print_bin_matrix(E);
    printf("Inverted matrix from GPU:\n");
    print_bin_matrix(inverted_gpu);

    printf("Time for CPU: %f\n", cpu_time_used);
    printf("Time for GPU: %f\n", gpu_time_used);


    bool correct = true;
    for (int i =0; i < A->rows*A->cols; i++) {
        if (E->data[i] != inverted_gpu->data[i]) {
            correct = false;
            break;
        }
    }

    printf("correctq: ");
    if (correct) printf("true\n");
    else (printf("false\n"));

    free(A);
    free(E);

    free(inverted_gpu);

    return 0;
}

#endif /* TEST_INVERSE_CPU_H */
