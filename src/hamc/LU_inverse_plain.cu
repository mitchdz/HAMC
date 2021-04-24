#ifndef LU_INVERSE_PLAIN_CU
#define LU_INVERSE_PLAIN_CU

#include "hamc_common.h"
#include "hamc_cpu_code.c"


// to be called with a single thread
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


// Forward Substitution to be used after LU Decomposition
//  A - input matrix (modified from LU decomposition)
//  B - identity matrix of size n
//  n - size of matrix A
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


// Backward Substition to be used after Forward Substitution
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



// This kernel swaps rows given an IPIV
//   A - matrix with rows to swap
//   IPIV - pivot vector
//   n - size of matrix A
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
// Update trailing matrix rows
//   A - matrix to update trailing rows
//   n - size of matrix A
//   k - row
// update trailing matrix C ^= A & B, 
// where A is A(k+1:n, k), B is A(k, k+1 : n), C is A(k+1: n, k+1:n)
//
// This kernel expects you to supply the col to be operated on.
__global__ void GF2_LU_decompose_update_trailing_row_index( HAMC_DATA_TYPE_t *A,
    int n, int k, int j)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Update trailing matrix C ^= A & B
    // where A is A(k+1:n, k), B is A(k, k+1 : n), C is A(k+1: n, k+1:n)
    int m = n - k - 1;
    if (tid < m) {
        //printf("tid: %d\n", tid);
        HAMC_DATA_TYPE_t *Arow = &A[(k + 1) * n + k];
        HAMC_DATA_TYPE_t *Brow = &A[k * n + k + 1];
        HAMC_DATA_TYPE_t *Crow = &A[(k + 1) * n + (k + 1)];
        Crow[tid * n + j] ^= Arow[tid * n] & Brow[j];
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
                return;
            }
        }
    }
}

// A is input matrix
// IPIV is integer pivot indeces vector should be sizeof(int)*A->rows,
// n is ld (latent dimension) should be A->rows or A->cols
__global__ void GF2_LU_decompose_pivot_row( HAMC_DATA_TYPE_t *A, int *IPIV, 
    int n, int k)
{
    bool verbose = true;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

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

        //GF2_LU_decompose_update_trailing_row
        //    <<<dimGrid, dimThreads, 0, stream1>>>(deviceA, A->rows, i);

        // above kernel times out if matrix is too large. 
        // Iterate through each row here.
        for (int j = 0; j < A->rows - i - 1; j++) { // rows
            GF2_LU_decompose_update_trailing_row_index
                <<<dimGrid, dimThreads, 0, stream1>>>(deviceA, A->rows, i, j);
        }
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
    GF2_Forward_substitute<<<dimGrid, dimThreads>>> 
        (deviceA, deviceB, A->rows);

    clock_t LU_forward_end = clock();
    double LU_forward_time = 
        ((double) (LU_forward_end - LU_forward_start))/ CLOCKS_PER_SEC;


    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));

    /******************** Backward Substitution *******************************/
    clock_t LU_backward_start = clock();

    if (verbose) printf("Performing Backward Substitution...\n");
    GF2_Backward_substitute<<<dimGrid, dimThreads>>>
        (deviceA, deviceB, A->rows);
    clock_t LU_backward_end = clock();
    double LU_backward_time = 
        ((double) (LU_backward_end - LU_backward_start))/ CLOCKS_PER_SEC;

    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));

    /******************** Final Swap ******************************************/
    clock_t LU_final_swap_start = clock();

    if (verbose) printf("Performing Final swap...\n");
    GF2_swap_rows<<<dimGrid, dimThreads>>>
        (deviceB, deviceIPIV, A->rows);
    clock_t LU_final_swap_end = clock();
    double LU_final_swap_time = 
        ((double) (LU_final_swap_end - LU_final_swap_start))/ CLOCKS_PER_SEC;

    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));


    if (verbose) printf("Done!\n");
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
         cudaGetErrorString(cudaerr));


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