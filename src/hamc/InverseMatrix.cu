
#include "hamc_cpu_code.c"
#include "hamc_common.h"


// Make sure to has as many threads as columns/rows
__global__ void make_GF2_identity_gpu_fast(HAMC_DATA_TYPE_t *A, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //for (int tid = 0; tid < n; tid++) {
    if (tid < n) {
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



// out should be an identity matrix
__global__ void binary_inverse_square_matrix_naive(HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = 0; i < cols; i++) {
        if (A[i*cols + i] == 1) {
            __syncthreads();
            for(int j = 0; j <  rows; j++) {
                if (i != j && A[j*cols + i] == 1) {
                    /* (i) notates the ith row of the matrix */
                    /* => denotes where the row which the output is stored in */

                    __syncthreads();

                    // add rows for both A and B
                    // each thread handles a column element
                    if (idx < cols) {
                        /* add rows to identity */
                        /* (i) ^ (j) => (j) */
                        B[j*cols + idx] = 
                            B[i*cols + idx] ^ B[j*cols + idx];

                        /* A is special, we only XOR from i to cols */
                        if (idx >= i) {
                            /* add rows to input */
                            /* (i) ^ (j) => (j) from i to cols */
                            A[j*cols + idx] = A[i*cols + idx] ^ A[j*cols + idx];
                        }
                    }
                }
            }
        }
        else {
            for(int k = i + 1; k < rows; k++) {
                if(A[k*cols + i] == 1) {
                    // for each column, XOR k and i, store into i
                    if (idx < cols) {
                        /* add rows to identity */
                        /* (k) ^ (i) => (i) */
                        //add_rows_cpu(B, k, i);
                        B[k*cols + i] = B[k*cols + idx] ^ B[i*cols + idx];

                        /* add rows to input */
                        /* (k) ^ (i) => (i) */
                        //add_rows_cpu(A, k, i);
                        B[k*cols + i] = B[k*cols + idx] ^ B[i*cols + idx];
                    }
                    i = i - 1;
                    break;
                }
            }
        }
    }

    // B is already the data we want.
    return;
}

bin_matrix run_inverse_kernel(bin_matrix A)
{
    bin_matrix C = mat_init_cpu(A->rows, A->cols);

    /* allocate device memory */
    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;
    cudaMalloc((void **) &deviceA, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));

    /* transfer host data to device */
    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);


    int numThreadsPerBlock = 1024;
    int numGrids = A->cols/numThreadsPerBlock + 1;

    dim3 dimgrid(numGrids);
    dim3 dimBlock(numThreadsPerBlock);

    printf("Starting Inverse matrix kernel...\n");


    make_GF2_identity_gpu_fast<<<dimgrid,dimBlock>>>(deviceB, A->rows);


    binary_inverse_square_matrix_naive<<<dimgrid, dimBlock>>>(deviceA, deviceB,
        A->rows, A->cols);


    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(C->data, deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);

    return C;
}
