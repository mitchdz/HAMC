#ifndef HAMC_SCRATCH_H
#define HAMC_SCRATCH_H


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
#include "../../src/hamc/LU_inverse_block.cu"


int main(int argc, char *argv[]){

    bool verbose = false;

    int flag=0;

    int n = 2;
    int p = 6;
    int N = p;
    int t = 10;
    int w = 30;
    int seed = 10;

    int opt;
    while ((opt = getopt(argc, argv, "n:")) != -1){
        switch(opt){
            case 'n':
                p = atoi(optarg);
                break;
        }
    }

    N = p;

    printf("Size of input matrix: %s%d%s\n", YELLOW, p, NC);

    printf("Generating QC_MDPC code...\n");
    mdpc code = qc_mdpc_init_cpu(n, p, t, w, seed);
    printf("Generating Binary Circulant Matrix...\n");
    bin_matrix invertible_matrix = make_matrix_cpu(
        code->p, code->p, 
        splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 
        1);
    printf("Generated test matrix\n");

    // Copy matrix in two test matrices
    bin_matrix extra_matrix = mat_init_cpu(p, p);
    bin_matrix extra_matrix2 = mat_init_cpu(p, p);
    for (int i =0; i < p*p; i++) {
        HAMC_DATA_TYPE_t temp = invertible_matrix->data[i];
        extra_matrix->data[i] = temp;
        extra_matrix2->data[i] = temp;
    }

    clock_t hamc_cpu_start = clock();

    bin_matrix cpu_sol = circ_matrix_inverse_cpu(extra_matrix);

    clock_t hamc_cpu_end = clock();
    double hamc_cpu_time_used =
        ((double) (hamc_cpu_end - hamc_cpu_start))/ CLOCKS_PER_SEC;

    // Print input and expected result
    if (true) {
        printf("\nInput matrix A:\n");
        print_bin_matrix(invertible_matrix);

        printf("\nExpected solution is:\n");
        print_bin_matrix(cpu_sol);
    }

    clock_t lu_gpu_start = clock();
    bin_matrix new_gpu_sol = inverse_GF2_LU_block_gpu(extra_matrix2);
    clock_t lu_gpu_end = clock();
    double lu_gpu_time_used = ((double) (lu_gpu_end - lu_gpu_start))/
        CLOCKS_PER_SEC;


    if (verbose) print_bin_matrix(new_gpu_sol);

    // check results
    for (int i = 0; i < N*N; i++) {
        if (new_gpu_sol->data[i] != cpu_sol->data[i]) {
            flag = 1;
            break;
        }
    }

    if(flag==0) 
        printf("correctq: true");
    else 
        printf("correctq: Failure\n");


    printf("\n");
    //printf("Time for LU Decomposition CPU code: %lf\n", lu_cpu_time_used);
    printf("Time for HAMC CPU code:             %lfs\n", hamc_cpu_time_used);
    printf("Time for LU Decomposition GPU code: %lfs\n", lu_gpu_time_used);


    printf("Speed difference: %.2lfX ", lu_gpu_time_used/hamc_cpu_time_used);
    if (lu_gpu_time_used > hamc_cpu_time_used)
        printf("slower\n");
    else
        printf("faster\n");

    free(invertible_matrix);
    free(extra_matrix);
    free(cpu_sol);
    free(new_gpu_sol);

    return 0;
}


#endif /* HAMC_SCRATCH_H */