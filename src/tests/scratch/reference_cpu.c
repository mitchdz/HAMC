#ifndef REFERENCE_CPU_C_H_N
#define REFERENCE_CPU_C_H_N

#include "../hamc/hamc_common.h"
#include "../hamc/hamc_cpu_code.c"

void print_bin_matrix(bin_matrix A)
{
    printf(" ");
    for ( int i = 0; i < A->rows; i++) {
        for ( int j = 0; j < A->cols; j++) {
            printf("%d ", A->data[i*A->cols + j]);
        }
        printf("\n ");
    }
}

void LU_GF2_find_max_cpu(int n, HAMC_DATA_TYPE_t *A, int *IPIV, int off) {
    int i;
    for (i = 0; i < n; i++) {
        if (A[i] == 1) {
            *IPIV = i + off;
            return;
        }
    }
}

void LU_GF2_swap_rows_cpu(int n, HAMC_DATA_TYPE_t *R1, HAMC_DATA_TYPE_t *R2) {
    int i;
    for (i = 0; i < n; i++) {
        HAMC_DATA_TYPE_t temp;
        temp = R1[i];
        R1[i] = R2[i];
        R2[i] = temp;
    }
}

void LU_GF2_update_trailing_cpu(int m , int n, HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C) {
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            C[i * n + j] ^= A[i * n] & B[j];
        }
    }
}

bin_matrix inverse_GF2_cpu(bin_matrix A)
{
    printf("\nCPU based inverse\n");

    int n = A->rows;

    int *IPIV = (int *)malloc(n*sizeof(int));

    /* LU decomposition */
    for (int k = 0; k < n; k++) {
        // Call 1st kernel: find the nearest ‘1’ on the subdiagonal column 
        // (column k, rows k:n), put the row number in IPIV[k]
        LU_GF2_find_max_cpu(n - k, &A->data[k * n + k], &IPIV[k], k);

        // Call 2nd kernel that swaps rows k and IPIV[k]

        if (k != IPIV[k]) {
            LU_GF2_swap_rows_cpu(n, &A->data[k * n], &A->data[IPIV[k] * n]);
        }

        // Call 3rd kernel: Update trailing matrix C ^= A & B, 
        // where A is A(k+1:n, k), B is A(k, k+1 : n), C is A(k+1: n, k+1:n)
        LU_GF2_update_trailing_cpu(n - k - 1,
            n - k - 1,
            &A->data[(k + 1) * n + k],
            &A->data[k * n + k + 1],
            &A->data[(k + 1) * n + (k + 1)]);
    }

    print_bin_matrix(A);
    printf("\nIPIV:\n");

    for (int i = 0; i < A->rows; i++) {
        printf("%d ", IPIV[i]);
    }
    printf("\n");


    /* Forward Backward Substitution */


    bin_matrix IA = mat_init_cpu(n, n);


    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            IA->data[i*n + j] =  IPIV[i] == j ? 1 : 0;

            for (int k = 0; k < i; k++) {
                IA->data[i*n + j] ^= A->data[i*n + k] & IA->data[k*n + j];
            }
        }

        for (int i = n - 1; i >= 0; i--) {
            for (int k = i + 1; k < n; k++) {
                IA->data[i*n + j] ^= A->data[i*n + k] & IA->data[k*n + j];
            }
        }
    }

    printf("\nsolution:\n");

    print_bin_matrix(IA);

    return IA;


}

#endif /* REFERENCE_CPU_C_H_N */