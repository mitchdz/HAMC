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



void run_find_max_kernel(bin_matrix A)
{
    HAMC_DATA_TYPE_t *deviceA;

    cudaMalloc((void **)
        &deviceA, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));

    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), 
        cudaMemcpyHostToDevice);

    int *deviceIPIV;
    cudaMalloc((void **) &deviceIPIV, A->rows * sizeof(int));

    GF2_LU_decompose_find_max_row<<<1,1>>>(deviceA, deviceIPIV, 0, 0);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));

    return;
}


int main(int argc, char *argv[]){

    bool verbose = true;

    int n = 2;
    int p = 512;
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

    bin_matrix invertible_matrix;

    mdpc code;

    code = qc_mdpc_init_cpu(n, p, t, w, seed);
    invertible_matrix = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 1);

    if (verbose) {
        printf("Input matrix size: %dx%d\n",
            invertible_matrix->rows, invertible_matrix->cols);
    }

    run_find_max_kernel(invertible_matrix);


    if (verbose) printf("Freeing allocated memory...\n");
    if (invertible_matrix != NULL) free(invertible_matrix);

    return 0;
}


#endif /* HAMC_SCRATCH_H */