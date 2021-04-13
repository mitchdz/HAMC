#ifndef HAMC_DATA_TYPE_t
#define HAMC_DATA_TYPE_t HAMC_DATA_TYPE_t
#endif

#include "hamc_cpu_code.c"


#define mat_element_gpu(mat, cols, row_idx, col_idx) \
          mat[row_idx * (cols) + col_idx]


//TODO: make device function to generate NxN identity matrix

__device__ int getBIndex(int i,  int j, int rows, int cols);
__device__ int getAIndex(int i, int j, int rows, int cols);


__global__ void binary_gaussian_elimination_with_pivot(HAMC_DATA_TYPE_t *in, HAMC_DATA_TYPE_t *out, int rows, int cols)
{
    __shared__ HAMC_DATA_TYPE_t A[16*2];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    /**
     * if "in" array is:
     * 1 0 1
     * 1 0 0
     * 1 1 1
     * then A shall look like:
     * left   right
     * half   half
     * 1 0 1  1 0 0
     * 1 0 0  0 1 0
     * 1 1 1  0 0 1
     * where the left half is "in" and the right half is the identity matrix
     **/
    /* store "in" to left half of A, and identity matrix to right half */
    if (idx == 0) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat_element_gpu(A,cols,i,j) = mat_element_gpu(in,cols,i,j);

                if (i == j) A[getBIndex(i, j, rows, cols)] = 1;
                else A[getBIndex(i, j, rows, cols)] = 0;
            }
        }
    }

    //// for each column k = 0 : cols
    //for (int k = 0; k < cols; k++) {
    //    //while a_11 = 0
    //    //    shiftup(cols-k+1, A)
    //    while (A[0] == 0) {
    //    }

    //    // A = eliminate(A);

    //}


    // write output to "out"


}


#define mat_size 16

// shared memory can only be one variable, therefore both of
// the matrix A and B need to be in one variable.
// This function simply offsets the address to be where the B
// matrix starts and then accesses the necessary address.
__device__ int getBIndex(int i,  int j, int rows, int cols)
{
    return rows*cols + i*cols + j;
}

__device__ int getAIndex(int i, int j, int rows, int cols)
{
    return i*cols + j;
}




// uses shared memory
// each thread handles a single column
__global__ void binary_inverse_square_matrix_naive(HAMC_DATA_TYPE_t *in, HAMC_DATA_TYPE_t *out, int rows, int cols)
{
    //int matrix_size = rows*cols;

    // we need to store two arrays
    // left half = input matrix
    // right half = output matrix
    __shared__ HAMC_DATA_TYPE_t A[16*2];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx == 0) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat_element_gpu(A,cols,i,j) = mat_element_gpu(in,cols,i,j);

                if (i == j) A[getBIndex(i, j, rows, cols)] = 1;
                else A[getBIndex(i, j, rows, cols)] = 0;
            }
        }
    }

    __syncthreads();

    if (idx == 0 ) {
        printf("KERNEL: rows: %d cols: %d\n", rows, cols);
        for (int i =0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%hu ", mat_element_gpu(A, cols, i, j));
            }
            printf("\n");
        }
    }

    /* wait for identity matrix to be created */
    __syncthreads();


    printf("Finished setting up matrices\n");

    for(int i = 0; i < cols; i++) {
        printf("i: %d\n", i);
        __syncthreads();
        //if(mat_element_gpu(A, cols, i, i) == 1) {
        if (A[getAIndex(i, i, rows, cols)] == 1) {
            __syncthreads();
            for(int j = 0; j <  rows; j++) {

                //printf("\ti=%d ,j=%d ,A[j,i]=%d, \n", i, j, mat_element_gpu(A, cols, j, i));
                printf("\ti=%d ,j=%d ,A[j,i]=%d, \n", i, j, 
                        A[getAIndex(j, i, rows, cols)]);
                __syncthreads();
                //if(i != j && mat_element_gpu(A, cols, j, i) == 1) {
                if (i != j && A[getAIndex(j, i, rows, cols)] == 1) {
                    printf("\t\t got into last if statement\n");
                    /* (i) notates the ith row of the matrix */
                    /* => denotes where the row which the output is stored in */

                    __syncthreads();

                    /* add rows for both A and B*/
                    //printf("i: %d j: %d\n", i, j );
                    for(int colid = 0; colid < cols; colid++) {
                        printf("\t\t\tcolid: %d from thread %d\n", colid, idx);
                        /* each thread handles adding a value in both A & B */
                        __syncthreads();
                        if (idx == colid) {

                            printf("\t\t\t\tidx: %d, colid: %d\n", idx, colid);

                            /* add rows to identity */
                            /* (i) ^ (j) => (j) */
                            //add_rows_new_cpu(B, i, j, 0, A->cols);
                            //mat_element_gpu(B, cols, j, colid) =
                            A[getBIndex(j, colid, rows, cols)] =
                                //(mat_element_gpu(B, cols, i, colid)
                                (A[getBIndex(i, colid, rows, cols)]
                                //^ mat_element_gpu(B, cols, j, colid));
                                ^ A[getBIndex(j, colid, rows, cols)]);

                            /* A is special, we only XOR from i to cols */
                            if (colid >= i) {
                                /* add rows to input */
                                /* (i) ^ (j) => (j) from i to cols */
                                //add_rows_new_cpu(A, i, j, i, A->cols);
                                mat_element_gpu(A, cols, j, colid) =
                                    (mat_element_gpu(A, cols, i, colid)
                                    ^ mat_element_gpu(A, cols, j, colid));

                            }
                            __syncthreads();
                        }
                        __syncthreads();
                    }
                    __syncthreads();
                }
                __syncthreads();
            }
            __syncthreads();
        }
        else {
            __syncthreads();
            for(int k = i + 1; k < rows; k++) {
                printf("k: %d\n", k);
                __syncthreads();
                if(mat_element_gpu(A, cols, k, i) == 1) {
                    __syncthreads();
                    // for each column, XOR k and i, store into i
                    for(int colid = 0; colid < cols; colid++) {
                        /* each thread handles adding a val in both A & B */
                        __syncthreads();
                        if (idx == colid) {
                            /* add rows to identity */
                            /* (k) ^ (i) => (i) */
                            //add_rows_cpu(B, k, i);
                            mat_element_gpu(A, cols, k, i) =
                                (mat_element_gpu(A, cols, k, colid)
                                ^ mat_element_gpu(A, cols, i, colid));

                             /* add rows to input */
                             /* (k) ^ (i) => (i) */
                             //add_rows_cpu(A, k, i);

                             //mat_element_gpu(B, cols, k, i) =
                             A[getBIndex(k, i, rows, cols)] =
                                //(mat_element_gpu(B, cols, k, colid)
                                (A[getBIndex(k, colid, rows, cols)]
                                //^ mat_element_gpu(B, cols, i, colid));
                                ^ A[getBIndex(i, colid, rows, cols)]);
                        }
                        __syncthreads();

                    }
                    __syncthreads();
                    i = i - 1;
                    __syncthreads();
                    break;
                }
                __syncthreads();
            }
            __syncthreads();
        }
        __syncthreads();
    }

    __syncthreads();

    //if (idx == 0) {
    //    for (int i = 0; i < rows; i++) {
    //        for (int j = 0; j < rows; j++) {
    //            printf("%hu ", mat_element_gpu(B, cols, i, j));
    //        }
    //        printf("\n");
    //    }
    //}

    __syncthreads();

    //write to output
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            //mat_element_gpu(out, cols, i, j) = mat_element_gpu(B, cols, i, j);
            mat_element_gpu(out, cols, i, j) = A[getBIndex(i, j, rows, cols)];
        }
    }

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

    printf("A from run_inverse_kernel:\n");
    for (int i =0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            printf("%hu ", mat_element_gpu(A->data, A->cols, i, j));
        }
        printf("\n");
    }



    printf("Starting Inverse matrix kernel...\n");

    // /* determine block and grid dimensions */
    //dim3 DimBlock(TRANSPOSE_TILE_WIDTH, TRANSPOSE_TILE_WIDTH, 1);
    //int x_blocks = ((A->rows - 1)/TRANSPOSE_TILE_WIDTH) + 1;
    //int y_blocks = ((A->cols - 1)/TRANSPOSE_TILE_WIDTH) + 1;
    //dim3 DimGrid(x_blocks, y_blocks, 1);


    binary_inverse_square_matrix_naive<<<1, 4>>>
        (deviceA, deviceB, A->rows, A->cols);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(C->data, deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);

    return C;
}
