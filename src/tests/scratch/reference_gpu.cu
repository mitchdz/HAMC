
#ifndef REFERENCE_GPU_CU
#define REFERENCE_GPU_CU

#include "../hamc/hamc_common.h"
#include "../hamc/hamc_cpu_code.c"



__global__ void make_GF2_identity_gpu(HAMC_DATA_TYPE_t *A, int n)
{
    //TODO: actually do this function

}



// A is input matrix
// IPIV is integer pivot indeces vector should be sizeof(int)*A->rows,
// n is ld (latent dimension) should be A->rows or A->cols
__global__ void GF2_square_inverse( HAMC_DATA_TYPE_t *A, 
    HAMC_DATA_TYPE_t *B, int n)
{
    bool verbose = false;


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

    if (tid == 0) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(i == j) {
                    B[i*n + j] = 1;
                }
                else {
                    B[i*n + j] = 0;
                }
            }
        }
    }

    __syncthreads();

    if (tid == 0) {
        // Forward
        for (int j = 0; j < n; j++) { // cols
            for (int i = j - 1; i >= 0; i--) { // rows from bottom to top
                B[i*n + j] = A[i*n + j];
                for (int k = i+1; k < j; k++) {
                    B[i*n + j] ^= B[k*n + j] & A[i*n + k];
                }
            }
        }

        // Backward
        for (int j = n - 1; j >= 0; j--) { // cols from right to left
            for (int i = 0; i < n; i++) { // rows from top to bottom
                //IA->data[i*n + j] = A->data[i*n + j];
                for (int k = j+1; k < n; k++) {
                    B[i*n + j] ^= B[i*n + k] & A[k*n + j];
                }
            }
        }
    }

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

    GF2_square_inverse<<<1, A->rows, A->rows*sizeof(int)>>> (deviceA, deviceB, A->rows);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(B->data, deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);

    return B;
}




#endif /* REFERENCE_GPU_CU */