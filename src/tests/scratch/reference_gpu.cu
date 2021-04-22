
#ifndef REFERENCE_GPU_CU
#define REFERENCE_GPU_CU

#include "../hamc/hamc_common.h"
#include "../hamc/hamc_cpu_code.c"
#include <cuda_device_runtime_api.h>


// to be called with a single threads
__global__ void make_GF2_identity_gpu(HAMC_DATA_TYPE_t *A, int n)
{
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
          if(i == j) {
              A[i*n + j] = 1;
          }
          else {
              A[i*n + j] = 0;
          }
        }
    }
}

// A is input matrix
// IPIV is integer pivot indeces vector should be sizeof(int)*A->rows,
// n is ld (latent dimension) should be A->rows or A->cols
__global__ void GF2_square_inverse( HAMC_DATA_TYPE_t *A, 
    HAMC_DATA_TYPE_t *B, int n)
{
    bool verbose = true;

    extern __shared__ int IPIV[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // TODO: Have a single thread make the identity matrix in the background
    // to hide the latency of generating the matrix

    // call a new kernel with one thread
    //make_GF2_identity_gpu<<< 1, 1>>>(B, n);

    for (int k = 0; k < n; k++) {
        // find max
        if (tid == 0) {
        //void LU_GF2_find_max_cpu(int n, HAMC_DATA_TYPE_t *A, int ld, int *IPIV, int off) {
        //        LU_GF2_find_max_cpu(n - k, &A->data[k * n + k], n, &IPIV[k], k);
            for (int i = 0; i < n - k; i++) {
                HAMC_DATA_TYPE_t *Arow = &A[k * n + k];
                if (Arow[i*n] == 1) {
                    //printf("IPIV[%d] = %d + %d\n", k, i, k);
                    IPIV[k] = i + k;
                    break;
                }
            }
        }

        // wait for thread 0 to find max and store in IPIV
        __syncthreads();
        

        //printf("k: %d, IPIV[%d] = %d\n", k, k, IPIV[k]);

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

        // wait for all threads to finish swapping column elements
        __syncthreads();

        // Update trailing matrix C ^= A & B
        // where A is A(k+1:n, k), B is A(k, k+1 : n), C is A(k+1: n, k+1:n)
        if (tid == 0) {
            int m = n - k - 1;
            for (int i = 0; i < m; i++) { // row
                for (int j = 0; j < m; j++) { // cols
                    HAMC_DATA_TYPE_t *Arow = &A[(k + 1) * n + k];
                    HAMC_DATA_TYPE_t *Brow = &A[k * n + k + 1];
                    HAMC_DATA_TYPE_t *Crow = &A[(k + 1) * n + (k + 1)];
                    Crow[i * n + j] ^= Arow[i * n] & Brow[j];
                }
            }
        }
        // wait for all threads up finish update trailing
        __syncthreads();
    }

    if (verbose) {
        if (tid == 0) {
            printf("\nIPIV from GPU:\n");
            for (int i = 0; i < n; i++) {
                printf("%d ", IPIV[i]);
            }
            printf("\n\n");
        }
    }

    // make B an identity matrix
    /*
    // have each thread handle a row
    if (tid < n) {
        for(int j = 0; j < n; j++) {
            if(tid == j) {
                B[tid*n + j] = 1;
            }
            else {
                B[tid*n + j] = 0;
            }
        }
    }
    */

    if (tid == 0) printf("\nMade identity matrix\n");

    __syncthreads();

    // Forward
    if (tid < n) { // cols
        for (int i = tid - 1; i >= 0; i--) { // rows from bottom to top
            B[i*n + tid] = A[i*n + tid];
            for (int k = i+1; k < tid; k++) {
                B[i*n + tid] ^= B[k*n + tid] & A[i*n + k];
            }
        }
    }

    __syncthreads();
    if (tid == 0) printf("\nForward subsititution done\n");

    // Backward
    for (int j = n - 1; j >= 0; j--) { // cols from right to left
        if (tid < n) { // rows from top to bottom
            //IA->data[i*n + j] = A->data[i*n + j];
            for (int k = j+1; k < n; k++) {
                B[tid*n + j] ^= B[tid*n + k] & A[k*n + j];
            }
        }
    }

    __syncthreads();
    if (tid == 0) printf("\nBackward subsititution done\n");

    for (int k = n - 1; k >= 0; k--) { // cols from right to left
        // swap cols
        //LU_GF2_swap_cols_cpu(n, &IA->data[k], &IA->data[k], n);
        
        // each thread handles a row element from C1 and C2
        if (tid < n) {
            HAMC_DATA_TYPE_t *C1 = &B[k];
            HAMC_DATA_TYPE_t *C2 = &B[IPIV[k]];
            HAMC_DATA_TYPE_t temp = C1[tid*n];
            C1[tid*n] = C2[tid*n];
            C2[tid*n] = temp;
        }
    }

    // Matrix B already has output, no need to copy the data
}


bin_matrix inverse_GF2_gpu(bin_matrix A)
{
    /* transpose so rows/cols flipped */
    bin_matrix B = mat_init_cpu(A->rows, A->cols);

    /* allocate device memory */
    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;
    cudaMalloc((void **) &deviceA, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));

    /* transfer host data to device */
    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);

    printf("Starting Inverse matrix kernel...\n");


    // total number of threads should be at least A->cols
    dim3 dimGrid = dim3(A->cols/1024 + 1, 1);
    dim3 dimThreads = dim3(1024, 1);

    cudaStream_t stream0;
    cudaStream_t stream1;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    make_GF2_identity_gpu<<<1,1,0,stream0>>>(deviceB, A->rows);
    GF2_square_inverse<<<dimGrid, dimThreads, A->rows*sizeof(int), stream1>>> (deviceA, deviceB, A->rows);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(B->data, deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);

    return B;
}




#endif /* REFERENCE_GPU_CU */