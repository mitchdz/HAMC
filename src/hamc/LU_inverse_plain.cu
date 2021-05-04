#ifndef LU_INVERSE_PLAIN_CU
#define LU_INVERSE_PLAIN_CU

#include "hamc_common.h"
#include "hamc_cpu_code.c"
#include "LU_inverse_plain_kernels.cu"
bin_matrix inverse_GF2_LU_gpu(bin_matrix A, bool verbose)
{
    // B is output matrix
    bin_matrix B = mat_init_cpu(A->rows, A->cols);

    clock_t LU_start = clock();

    /* allocate device memory */
    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;

    int *hostIPIV = (int *)malloc(A->rows*sizeof(int));

    int *deviceIPIV;
    cudaMalloc((void **)
        &deviceA, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **)
        &deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceIPIV, A->rows * sizeof(int));

    /* transfer host data to device */
    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), 
        cudaMemcpyHostToDevice);

    if (verbose) printf("Starting Inverse matrix kernel...\n");

    // total number of threads should be at least A->cols
    int numThreadsPerBlock = 1024;

    int numGrids = A->cols/numThreadsPerBlock + 1;

    if (verbose) {
        printf("\t# threadBlocks: %s%d%s\n", YELLOW, numGrids, NC);
        printf("\t# threads per block: %s%d%s\n", YELLOW, numThreadsPerBlock,
            NC);
        printf("\tTotal threads: %s%d%s\n", YELLOW,numGrids*numThreadsPerBlock,
            NC);
    }

    dim3 dimGrid = dim3(numGrids, 1);
    dim3 dimThreads = dim3(numThreadsPerBlock);

    cudaStream_t stream0;
    cudaStream_t stream1;

    cudaError_t cudaerr;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // Streaming identity matrix generation to hide the latency
    // while we do LU decomposition
    make_GF2_identity_gpu<<<1,1,0,stream0>>>(deviceB, A->rows);

    /******************** LU decomposition ************************************/
    if (verbose) printf("Performing LU Decomposition...\n");
    clock_t LU_decompose_start = clock();

    // Unfortunately this has to be asynchronous.
    for (int i = 0; i < A->rows; i++) {
        GF2_LU_decompose_find_max_row<<<1, 1, 0, stream1>>> 
            (deviceA, deviceIPIV, A->rows, i);

        GF2_LU_decompose_pivot_row<<<dimGrid, dimThreads, 0, stream1>>> 
            (deviceA, deviceIPIV, A->rows, i);

        GF2_LU_decompose_update_trailing_row
            <<<dimGrid, dimThreads, 0, stream1>>>(deviceA, A->rows, i);
    }
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    clock_t LU_decompose_end = clock();
    double LU_decompose_time = 
        ((double) (LU_decompose_end - LU_decompose_start))/ CLOCKS_PER_SEC;


    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));

    /******************** Forward Substitution ********************************/
    clock_t LU_forward_start = clock();
    if (verbose) printf("Performing Forward Substitution...\n");
    GF2_Forward_substitute<<<dimGrid, dimThreads>>> (deviceA, deviceB, A->rows);


    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));
 
    clock_t LU_forward_end = clock();
    double LU_forward_time = 
        ((double) (LU_forward_end - LU_forward_start))/ CLOCKS_PER_SEC;


    /******************** Backward Substitution *******************************/
    clock_t LU_backward_start = clock();
    if (verbose) printf("Performing Backward Substitution...\n");
    GF2_Backward_substitute<<<dimGrid, dimThreads>>> (deviceA, deviceB, A->rows);

    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));


    clock_t LU_backward_end = clock();
    double LU_backward_time = 
        ((double) (LU_backward_end - LU_backward_start))/ CLOCKS_PER_SEC;

    /******************** Final Swap ******************************************/
    clock_t LU_final_swap_start = clock();
    if (verbose) printf("Performing Final swap...\n");
    GF2_swap_cols<<<dimGrid, dimThreads>>>(deviceB, deviceIPIV, A->rows);

    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));

    clock_t LU_final_swap_end = clock();
    double LU_final_swap_time = 
        ((double) (LU_final_swap_end - LU_final_swap_start))/ CLOCKS_PER_SEC;
    if (verbose) printf("Done!\n");

    cudaMemcpy(B->data, deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), 
        cudaMemcpyDeviceToHost);

    clock_t LU_end = clock();
    double LU_time = 
        ((double) (LU_end - LU_start))/ CLOCKS_PER_SEC;

    if (verbose) {
        printf("Total time for LU inverse (GPU): %.7lf\n", LU_time);
        printf("\tLU decomposition:          %.7lf - %.2lf%%\n", 
            LU_decompose_time, 100*(LU_decompose_time/LU_time));
        printf("\tForward Substitution:      %.7lf - %.2lf%%\n",
            LU_forward_time, 100*(LU_forward_time/LU_time));
        printf("\tBackward Substitution:     %.7lf - %.2lf%%\n",
            LU_backward_time, 100*(LU_backward_time/LU_time));
        printf("\tFinal Swap:                %.7lf - %.2lf%%\n",
            LU_final_swap_time, 100*(LU_final_swap_time/LU_time));
    }
    cudaFree(deviceA);
    cudaFree(deviceB);

    free(hostIPIV);

    return B;
}


#endif /* REFERENCE_GPU_CU */