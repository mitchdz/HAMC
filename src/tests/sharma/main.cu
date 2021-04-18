#ifndef HAMC_SCRATCH_H
#define HAMC_SCRATCH_H

#include "../../src/hamc/hamc_cpu_code.c"

void printMatrix(uint8_t *mat, int n);


__global__ void fixRow(HAMC_DATA_TYPE_t *matrix, int size, int rowId)
{
    // The ith row of the matrix
    __shared__ HAMC_DATA_TYPE_t Ri[512];

    // The diagonal element for ith row
    __shared__ HAMC_DATA_TYPE_t Aii;

    int colId = threadIdx.x;
    Ri[colId] = matrix[size * rowId + colId];
    Aii = matrix[size * rowId + colId];
    __syncthreads();

    // Divide the whole row by the diagonal element making sure it is not 0
    Ri[colId] ^= Aii;
    matrix[size * rowId + colId] = Ri[colId];
}

__global__ void fixColumn(HAMC_DATA_TYPE_t *matrix, int size, int colId)
{
    int i = threadIdx.x;
    int j = blockIdx.x;

    // The colId column
    __shared__ HAMC_DATA_TYPE_t col[512];

    // The jth element of the colId row
    __shared__ HAMC_DATA_TYPE_t AColIdj;

    // The jth column
    __shared__ HAMC_DATA_TYPE_t colj[512];

    col[i] = matrix[i * size + colId];

    if (col[i] != 0) {
        colj[i] = matrix[i * size + j];
        AColIdj = matrix[colId * size + j];
        if (i != colId) {
            //colj[i]  = colj[i] - AColIdj * col[i];
            colj[i]  ^= AColIdj & col[i];
        }
        matrix[i * size + j] = colj[i];
    }
}



int getIndex(int cols, int row, int col)
{
    return row*cols + col;
}


int main()
{
    const int n = 4;
    // creating input
    uint8_t *h_A = new uint8_t[n*n];

    bin_matrix CPU = mat_init_cpu(n,n);

    uint8_t val;
    int seed = 11;
    // create random nxn binary matrix
    srand(seed);
    for ( int i = 0; i < n*2; i++) {
        val = rand() %2;
        h_A[i] = val;
        CPU->data[i] = val;
    }

    printf("h_A matrix:\n");
    printMatrix(h_A,n);
    printf("\n");


    /* copy host matrix to GPU */
    HAMC_DATA_TYPE_t *d_matrix;

    cudaMalloc((void **) &d_matrix, n * n * sizeof(HAMC_DATA_TYPE_t));

    cudaMemcpy(d_matrix, h_A,
            n*n*sizeof(HAMC_DATA_TYPE_t),
            cudaMemcpyHostToDevice);

    int j = 0;
    while (j < n) {
        // Find k where matrix[k][j] is not 0
        for (int k = 0; k < CPU->rows; k++) {
            if (h_A[getIndex(n, k, j)] == 1) {
                //fix row
                fixRow<<<1,n>>>(d_matrix, n, k);

                //fix column
                fixColumn<<<1,n>>>(d_matrix, n, j);
            }
        }
        j++;
    }


    bin_matrix h_B = mat_init_cpu(n,n);


    cudaMemcpy(h_B->data, d_matrix, n*n*sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);

    printf("GPU output matrix:\n");
    printMatrix(h_B->data,n);
    printf("\n");

    bin_matrix CPU_out = circ_matrix_inverse_cpu(CPU);

    printf("CPU output matrix:\n");
    printMatrix(CPU_out->data,n);
    printf("\n");

    free(h_A);
    free(h_B);
    free(CPU);
    free(CPU_out);


    return 0;
}


void printMatrix(uint8_t *mat, int n)
{
    for ( int i = 0; i < n; i++) {
        for ( int j = 0; j < n; j++) {
            printf("%d  ", mat[i*n+j]);
        }
        printf("\n");
    }
}


#endif /* HAMC_SCRATCH_H */
