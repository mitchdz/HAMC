#ifndef LU_INVERSE_BLOCK_CU
#define LU_INVERSE_BLOCK_CU

#include "hamc_common.h"
#include "hamc_cpu_code.c"

#include "LU_inverse_plain.cu"

#define LU_BLOCKING_SIZE 64

bin_matrix inverse_GF2_LU_block_gpu(bin_matrix A)
{
    // B is output matrix
    bin_matrix B = mat_init_cpu(A->rows, A->cols);

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

    printf("Starting Inverse matrix kernel...\n");

    // total number of threads should be at least A->cols
    //int numThreadsPerBlock = 1024;



    int n = A->rows;

    printf("Performing blocking LU decomposition...\n");
    /* blocking LU decomposition */
    for (int j = 0; j < n; j+= LU_BLOCKING_SIZE) {
        // 1) Factorize the panel A(j:n, j:j+nb)

        //IPIV - Integer Pivot Index Vector
        // Save pivots in IPIV(j:j+nb)

        // 2) apply row swaps to the left and right of the panel, 
        //    that is, interchange rows A(i, 0:j) with A(IPIV:i, 0:j) 
        //    and rows A(i, j+nb:n) with A(IPIV(i), j+nb:n), i = j .. j+nb-1


        // 3) compute the block row to the right of the panel: 
        //    solve L * X = U for X where L is lower triangular 
        //    kept in A(j:j+nb, j:j+nb), U is A(j:j+nb, j+nb:n) 
        //    with X overwriting U



        // 4) update the trailing submatrix D = D - L * U 
        //    where L is A(j+nb:n, j:j+nb), 
        //          U is A(j:j+nb, j+nb:n) 
        //      and D is A(j+nb:n, j+nb:n)
    }



    // Forward Substiution


    // Backward Substitution



    // Final Column Swap





    cudaFree(deviceA);
    cudaFree(deviceB);

    free(hostIPIV);

    return B;
}





#endif /* LU_INVERSE_BLOCK_CU */