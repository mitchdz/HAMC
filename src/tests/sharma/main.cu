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



int main()
{
    const int n = 4;
    // creating input
    uint8_t *h_A = new uint8_t[n*n];

    bin_matrix CPU = mat_init_cpu(n,n);

    uint8_t val;
    int seed = 10;
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


    int j = 0;
    while (j < n) {




        j++;
    }




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
