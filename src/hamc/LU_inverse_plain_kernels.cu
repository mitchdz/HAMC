
#ifndef LU_INVERSE_PLAIN_KERNELS_CU
#define LU_INVERSE_PLAIN_KERNELS_CU

#include "hamc_common.h"

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

// Forward Substitution to be used after LU Decomposition
//  A - input matrix (modified from LU decomposition)
//  B - identity matrix of size n
//  n - size of matrix A
__global__ void GF2_Forward_substitutev2(HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) { // cols
        for (int i = 0 ; i <= tid - 1; i++) { // rows
            int row = tid - 1 - i;
            B[row*n + tid] = A[row*n + tid];
            for (int k = row+1; k < tid; k++) {
                B[row*n + tid] ^= B[k*n + tid] & A[row*n + k];
            }
        }
    }
}

// Forward Substitution to be used after LU Decomposition
//  A - input matrix (modified from LU decomposition)
//  B - identity matrix of size n
//  n - size of matrix A
//  i - row to operate on
//  j - column to operate on
// call GF2_Forward_substitute_element_store before this w/ same parameters
__global__ void GF2_Forward_substitute_element(HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, int n, int i, int j)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;


    
    if (tid == 0) {
        for (int k = i+1; k < j; k++) {
            B[i*n + j] ^= B[k*n + j] & A[i*n + k];
        }
    }
    

    // example code below:
    //   for (int k = i+1; k < j; k++) {
    //      B[i*n + j] ^= B[k*n + j] & A[i*n + k];
    //   }


    /*
    if (tid == 0) {
        printf("i: %d j: %d\n", i, j);
    }

    if ((tid < j) && (tid >= i + 1)) {
        printf("tid: %d\n",tid);
        B[i*n + j] ^= B[tid*n + j] & A[i*n + tid];
    }
    */


}

// Forward Substitution to be used after LU Decomposition
//  A - input matrix (modified from LU decomposition)
//  B - identity matrix of size n
//  n - size of matrix A
//  i - row to operate on
//  j - column to operate on
__global__ void GF2_Forward_substitute_element_store(HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, int n, int i, int j)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        B[i*n + j] = A[i*n + j];
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

// Backward Substition to be used after Forward Substitution
__global__ void GF2_Backward_substitutev2(HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int j = 0; j <= n - 1; j++) { // cols from right to left
        if (tid < n) { // rows from top to bottom
            int col = n - 1 - j;
            //IA->data[i*n + j] = A->data[i*n + j];
            for (int k = col+1; k < n; k++) {
                B[tid*n + col] ^= B[tid*n + k] & A[k*n + col];
            }
        }
    }
}

// Backward Substition to be used after Forward Substitution
__global__ void GF2_Backward_substitute_row(HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, int n, int j)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) { // rows from top to bottom
        int col = n - 1 - j;
        //IA->data[i*n + j] = A->data[i*n + j];
        for (int k = col+1; k < n; k++) {
            B[tid*n + col] ^= B[tid*n + k] & A[k*n + col];
        }
    }
}


// This kernel swaps cols given an IPIV
//   A - matrix with cols to swap
//   IPIV - pivot vector
//   n - size of matrix A
__global__ void GF2_swap_cols(HAMC_DATA_TYPE_t *A, int *IPIV, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int k = n - 1; k >= 0; k--) { // cols from right to left        
        if (tid < n) {
            HAMC_DATA_TYPE_t *C1 = &A[k];
            HAMC_DATA_TYPE_t *C2 = &A[IPIV[k]];
            HAMC_DATA_TYPE_t temp = C1[tid*n];
            C1[tid*n] = C2[tid*n];
            C2[tid*n] = temp;
        }
    }
}

// This kernel swaps cols given an IPIV
//   A - matrix with cols to swap
//   IPIV - pivot vector
//   n - size of matrix A
__global__ void GF2_swap_colsv2(HAMC_DATA_TYPE_t *A, int *IPIV, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int k = 0; k <= n - 1; k++) { // cols from right to left  
        int row = n - 1 - k;
        if (tid < n) {
            HAMC_DATA_TYPE_t *C1 = &A[row];
            HAMC_DATA_TYPE_t *C2 = &A[IPIV[row]];
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

#endif /* LU_INVERSE_PLAIN_KERNELS_CU */