#ifndef HAMC_SCRATCH_H
#define HAMC_SCRATCH_H


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


#include "reference_cpu.c"
#include "inverse_gpu.cu"



int main(int argc, char *argv[]){

    bool verbose = true;

    printf("Scratch test\n");
    int flag=0;

    //bin_matrix cpu_raw = mat_init_cpu(N,N);
    //bin_matrix gpu_raw = mat_init_cpu(N,N);

    //srand(10);
    //fill the arrays 'a' on the CPU
    //for ( i = 0; i <= (N*N); i++) {
    //    HAMC_DATA_TYPE_t val = ((rand()%2));
    //    cpu_raw->data[i] = val;
    //    gpu_raw->data[i] = val;
    //}

    int n = 2;
    int p = 8;
    int N = p;
    int t = 10;
    int w = 30;
    int seed = 10;

    printf("Generating QC_MDPC code...\n");
    mdpc code = qc_mdpc_init_cpu(n, p, t, w, seed);
    printf("Generating Invertible Matrix...\n");
    bin_matrix invertible_matrix = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 1);
    bin_matrix extra_matrix = mat_init_cpu(p, p);

    printf("Generated test matrix\n");

    for (int i =0; i < p*p; i++) {
        extra_matrix->data[i] = invertible_matrix->data[i];
    }

    bin_matrix cpu_sol = circ_matrix_inverse_cpu(extra_matrix);

    if (verbose) {
        printf("\nInput matrix:\n");
        print_bin_matrix(invertible_matrix);

        printf("\nExpected solution is:\n");
        print_bin_matrix(cpu_sol);
    }


    bin_matrix new_cpu_sol = inverse_GF2_cpu(invertible_matrix);

    //bin_matrix gpu_sol = inverse_GF2_gpu(invertible_matrix);

    // check results
    for (int i = 0; i < N*N; i++) {
        if (new_cpu_sol->data[i] != cpu_sol->data[i]) {
            flag = 1;
            break;
        }
    }

    if(flag==0) printf("correctq: Correct");
    else printf("correctq: Failure\n");


    free(invertible_matrix);
    free(extra_matrix);
    free(cpu_sol);
    free(new_cpu_sol);

    return 0;
}


#endif /* HAMC_SCRATCH_H */