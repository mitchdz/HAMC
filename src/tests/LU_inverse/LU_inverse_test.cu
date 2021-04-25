#ifndef HAMC_SCRATCH_H
#define HAMC_SCRATCH_H

#include <wb.h>

#include <bits/getopt_core.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>

#include <sys/time.h>


#include "../../src/hamc/hamc_cpu_code.c"
#include "../../src/hamc/LU_inverse_plain.cu"


#include "LU_inverse_plain_cpu.c"


bin_matrix read_file_store_bin_matrix(const char *inputFile)
{
    bin_matrix A;

    int numRowsA, numColsA;

    printf("Reading %s\n", inputFile);
    float *floatTemp = (float *)wbImport(inputFile, &numRowsA, &numColsA);

    HAMC_DATA_TYPE_t *Adata = (HAMC_DATA_TYPE_t *)malloc(numRowsA*numColsA *
        sizeof(HAMC_DATA_TYPE_t));
    for(int i = 0; i < numRowsA * numColsA; i++){
        Adata[i] = (HAMC_DATA_TYPE_t)floatTemp[i];
    }

    A = mat_init_cpu(numRowsA, numColsA);
    A->data = Adata;

    free(floatTemp);
    return A;
}



int main(int argc, char *argv[]){

    bool verbose = true;

    int flag=0;

    int n = 2;
    int p = 6;
    int N = p;
    int t = 10;
    int w = 30;
    int seed = 10;

    bin_matrix cpu_sol = NULL, invertible_matrix;
    bin_matrix extra_matrix, extra_matrix2, new_cpu_sol = NULL, new_gpu_sol;

    mdpc code;
    clock_t hamc_cpu_start, hamc_cpu_end;
    double hamc_cpu_time_used;

    clock_t lu_cpu_start, lu_cpu_end;
    double lu_cpu_time_used;

    clock_t lu_gpu_start, lu_gpu_end;
    double lu_gpu_time_used;

    char *inputFile = NULL;
    char *expectedFile = NULL;


    bool cpu_exec = false;

    int opt;
    while ((opt = getopt(argc, argv, "n:i:e:cvs")) != -1){
        switch(opt){
            case 'n':
                p = atoi(optarg);
                break;
            case 'i':
                inputFile = strdup(optarg);
                break;
            case 'e':
                expectedFile = strdup(optarg);
                break;
            case 'c':
                cpu_exec = true;
                break;
            case 'v':
                verbose = true;
                break;
            case 's':
                verbose = false;
                break;
        }
    }

    // set N to p if it is changed from the user.
    N = p;

    if (!inputFile || !expectedFile) {
        if (verbose) {
            printf("No input or expected matrix, generating matrix...\n");
            printf("Generating QC_MDPC code...\n");
        }
        code = qc_mdpc_init_cpu(n, p, t, w, seed);
        if (verbose) printf("Generating Binary Circulant Matrix...\n");
        invertible_matrix = make_matrix_cpu(
            code->p, code->p, 
            splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 
            1);
        if (verbose) printf("Generated test matrix\n");
        cpu_exec = true;
    }
    else {
        // Get input and expected solution

        if (verbose) {
            printf("input file: %s\n", inputFile);
            printf("solution file: %s\n", expectedFile);
        }

        invertible_matrix = read_file_store_bin_matrix(inputFile);
        if (!cpu_exec)
            cpu_sol = read_file_store_bin_matrix(expectedFile);
    }

    if (verbose) {
        printf("Input matrix size: %dx%d\n",
            invertible_matrix->rows, invertible_matrix->cols);
    }

    p = invertible_matrix->rows;

    // Copy matrix in two test matrices
    extra_matrix = mat_init_cpu(p, p);
    extra_matrix2 = mat_init_cpu(p, p);
    for (int i =0; i < p*p; i++) {
        HAMC_DATA_TYPE_t temp = invertible_matrix->data[i];
        extra_matrix->data[i] = temp;
        extra_matrix2->data[i] = temp;
    }

    hamc_cpu_start = clock();

    if (cpu_exec) {
        if (verbose) printf("Performing CPU based execution...\n");
        cpu_sol = circ_matrix_inverse_cpu(extra_matrix);
        if (verbose) printf("Done\n");
    }


    hamc_cpu_end = clock();
    hamc_cpu_time_used =
        ((double) (hamc_cpu_end - hamc_cpu_start))/ CLOCKS_PER_SEC;

    // Print input and expected result
    if (verbose) {
        printf("\nInput matrix A:\n");
        if (invertible_matrix->rows < 60) print_bin_matrix(invertible_matrix);

        printf("\nExpected solution is:\n");
        if (cpu_sol->rows < 60) print_bin_matrix(cpu_sol);
        printf("\n");
    }

    lu_cpu_start = clock();
    //new_cpu_sol = inverse_GF2_cpu(invertible_matrix, verbose);
    lu_cpu_end = clock();
    lu_cpu_time_used = ((double) (lu_cpu_end - lu_cpu_start))/ CLOCKS_PER_SEC;


    lu_gpu_start = clock();
    new_gpu_sol = inverse_GF2_LU_gpu(extra_matrix2, verbose);
    lu_gpu_end = clock();
    lu_gpu_time_used = ((double) (lu_gpu_end - lu_gpu_start))/ CLOCKS_PER_SEC;

    if (verbose) printf("\nOutput from GPU:\n");
    if (verbose && new_gpu_sol->rows < 60) print_bin_matrix(new_gpu_sol);

    // check results
    for (int i = 0; i < N*N; i++) {
        if (new_gpu_sol->data[i] != cpu_sol->data[i]) {
            flag = 1;
            break;
        }
    }


    if(flag==0) 
        printf("correctq: %strue%s", GREEN, NC);
    else 
        printf("correctq: %sfalse%s\n", RED, NC);


    printf("\n");



    if (verbose) {
        if (!cpu_exec) {
            printf("Time for LU Decomposition GPU code: %lf s\n",
                lu_gpu_time_used);
        }

        if (cpu_exec) {
            printf("Time for HAMC CPU code: %lf s\n", 
                hamc_cpu_time_used);
            //printf("Speed difference (GPU LU vs. CPU gauss jordan): %.2lfX ", 
            //    lu_gpu_time_used/hamc_cpu_time_used);

            if (lu_gpu_time_used > hamc_cpu_time_used)
                printf("slower\n");
            else
                printf("faster\n");
        }
    }

    if (verbose) printf("Freeing allocated memory...\n");
    if (invertible_matrix != NULL) free(invertible_matrix);
    if (extra_matrix != NULL) free(extra_matrix);
    if (cpu_sol != NULL) free(cpu_sol);
    if (new_cpu_sol != NULL) free(new_cpu_sol);
    if (new_gpu_sol != NULL) free(new_gpu_sol);

    return 0;
}


#endif /* HAMC_SCRATCH_H */