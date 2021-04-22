
#ifndef REFERENCE_GPU_CU
#define REFERENCE_GPU_CU

#include "../hamc/hamc_common.h"
#include "../hamc/hamc_cpu_code.c"
#include <cuda_device_runtime_api.h>


// to be called with a single threads
__global__ void make_GF2_identity_gpu(HAMC_DATA_TYPE_t *A, int n)
{
    //int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = 0; tid < n; tid++) {
        for(int j = 0; j < n; j++) {
          if(tid == j) {
              A[tid*n + j] = 1;
          }
          else {
              A[tid*n + j] = 0;
          }
        }
    }
}

__global__ void GF2_Forward_substitute(HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) { // cols
        for (int i = tid - 1; i >= 0; i--) { // rows from bottom to top
            B[i*n + tid] = A[i*n + tid];
            for (int k = i+1; k < tid; k++) {
                B[i*n + tid] ^= B[k*n + tid] & A[i*n + k];
            }
        }
    }
}

__global__ void GF2_Backward_substitute(HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int j = n - 1; j >= 0; j--) { // cols from right to left
        if (tid < n) { // rows from top to bottom
            //IA->data[i*n + j] = A->data[i*n + j];
            for (int k = j+1; k < n; k++) {
                B[tid*n + j] ^= B[tid*n + k] & A[k*n + j];
            }
        }
    }
}


__global__ void GF2_swap_rows(HAMC_DATA_TYPE_t *A, int *IPIV, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int k = n - 1; k >= 0; k--) { // cols from right to left        
        // each thread handles a row element from C1 and C2
        if (tid < n) {
            HAMC_DATA_TYPE_t *C1 = &A[k];
            HAMC_DATA_TYPE_t *C2 = &A[IPIV[k]];
            HAMC_DATA_TYPE_t temp = C1[tid*n];
            C1[tid*n] = C2[tid*n];
            C2[tid*n] = temp;
        }
    }
}


__global__ void GF2_LU_decompose_update_trailing_row( HAMC_DATA_TYPE_t *A,
    int n, int k)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Update trailing matrix C ^= A & B
    // where A is A(k+1:n, k), B is A(k, k+1 : n), C is A(k+1: n, k+1:n)
    int m = n - k - 1;
    if (tid < m) {
        for (int j = 0; j < m; j++) { // cols
            HAMC_DATA_TYPE_t *Arow = &A[(k + 1) * n + k];
            HAMC_DATA_TYPE_t *Brow = &A[k * n + k + 1];
            HAMC_DATA_TYPE_t *Crow = &A[(k + 1) * n + (k + 1)];
            Crow[tid * n + j] ^= Arow[tid * n] & Brow[j];
        }
    }
}


// A is input matrix
// IPIV is integer pivot indeces vector should be sizeof(int)*A->rows,
// n is ld (latent dimension) should be A->rows or A->cols
__global__ void GF2_LU_decompose_find_max_row( HAMC_DATA_TYPE_t *A, int *IPIV, 
    int n, int k)
{
    bool verbose = true;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // find max
    if (tid == 0) {
        for (int i = 0; i < n - k; i++) {
            HAMC_DATA_TYPE_t *Arow = &A[k * n + k];
            if (Arow[i*n] == 1) {
                IPIV[k] = i + k;
                break;
            }
        }
    }

    // wait for thread 0 to find max and store in IPIV
    __syncthreads();

    // if pivot row is changed for kth row, swap row k with pivot row
    if (k != IPIV[k]) {
        // have each thread handle a separate column element
        // Make sure you have at least as many threads as n!!!
        if (tid < n) {
            HAMC_DATA_TYPE_t *R1 = &A[k * n]; // kth row
            HAMC_DATA_TYPE_t *R2 = &A[IPIV[k] * n]; // pivot row
            HAMC_DATA_TYPE_t temp = R1[tid];
            R1[tid] = R2[tid];
            R2[tid] = temp;
        }
    }
}


bin_matrix inverse_GF2_gpu(bin_matrix A)
{
    // B is output matrix
    bin_matrix B = mat_init_cpu(A->rows, A->cols);

    /* allocate device memory */
    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;
    int *deviceIPIV;
    cudaMalloc((void **)
        &deviceA, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **)
        &deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceIPIV, A->rows * sizeof(int));


    /* transfer host data to device */
    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), 
        cudaMemcpyHostToDevice);

    printf("Starting Inverse matrix kernel...\n");


    // total number of threads should be at least A->cols
    dim3 dimGrid = dim3(A->cols/1024 + 1, 1);
    dim3 dimThreads = dim3(1024, 1);

    cudaStream_t stream0;
    cudaStream_t stream1;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    make_GF2_identity_gpu<<<1,1,0,stream0>>>(deviceB, A->rows);

    printf("Performing LU Decomposition...\n");

    // Unfortunately this has to be asynchronous
    for (int i = 0; i < A->rows; i++) {
            GF2_LU_decompose_find_max_row<<<dimGrid, dimThreads, 0, stream1>>> 
                (deviceA, deviceIPIV, A->rows, i);

            GF2_LU_decompose_update_trailing_row<<<dimGrid, 
                dimThreads, 0, stream1>>>
                (deviceA, A->rows, i);
    }
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    cudaDeviceSynchronize();

    printf("Performing Forward Substitution...\n");
    GF2_Forward_substitute<<<dimGrid, dimThreads>>> 
        (deviceA, deviceB, A->rows);

    cudaDeviceSynchronize();


    printf("Performing Backward Substitution...\n");
    GF2_Backward_substitute<<<dimGrid, dimThreads>>>
        (deviceA, deviceB, A->rows);

    cudaDeviceSynchronize();


    printf("Performing Final swap...\n");
    GF2_swap_rows<<<dimGrid, dimThreads>>>
        (deviceB, deviceIPIV, A->rows);

    cudaDeviceSynchronize();


    printf("Done!\n");
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));

    cudaMemcpy(B->data, deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), 
        cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);

    return B;
}




#endif /* REFERENCE_GPU_CU */