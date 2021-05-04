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



void run_find_forward_kernel(bin_matrix A)
{

    int numThreadsPerBlock = 1024;
    int numGrids = A->cols/numThreadsPerBlock + 1;

    dim3 dimgrid = dim3(numGrids);
    dim3 dimBlock = dim3(numThreadsPerBlock);


    HAMC_DATA_TYPE_t *deviceA;

    cudaMalloc((void **)
        &deviceA, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));


    HAMC_DATA_TYPE_t *deviceB;
    cudaMalloc((void **)
        &deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));

    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), 
        cudaMemcpyHostToDevice);


    make_GF2_identity_gpu<<<1,1>>>(deviceB, A->rows);


    GF2_Forward_substitute<<<dimgrid, dimBlock>>> 
        (deviceA, deviceB, A->rows);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));

    return;
}



int main(int argc, char *argv[]){

    printf("Performing LU_inverse_forward_subsitution test\n");

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
    invertible_matrix = make_matrix_cpu(
        code->p, code->p, 
        splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 
        1);

    if (verbose) {
        printf("Input matrix size: %dx%d\n",
            invertible_matrix->rows, invertible_matrix->cols);
    }

    run_find_forward_kernel(invertible_matrix);


    if (verbose) printf("Freeing allocated memory...\n");
    if (invertible_matrix != NULL) free(invertible_matrix);

    return 0;
}


#endif /* HAMC_SCRATCH_H */