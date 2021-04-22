#ifndef REFERENCE_CPU_C_H_N
#define REFERENCE_CPU_C_H_N

#include "../hamc/hamc_common.h"
#include "../hamc/hamc_cpu_code.c"

void print_bin_matrix(bin_matrix A)
{
    printf("");
    for ( int i = 0; i < A->rows; i++) {
        printf("%d  ", i);
        for ( int j = 0; j < A->cols; j++) {
            printf("%d ", A->data[i*A->cols + j]);
        }
        printf("\n");
    }
}

// LU_GF2_find_max_cpu(n - k, &A->data[k * n + k], n, &IPIV[k], k);
void LU_GF2_find_max_cpu(int n, HAMC_DATA_TYPE_t *A, int ld, int *IPIV, int off) {
    int i;
    for (i = 0; i < n; i++) {
        if (A[i*ld] == 1) {
            *IPIV = i + off;
            return;
        }
    }
    printf("\nMatrix is singular\n");
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

void LU_GF2_swap_cols_cpu(int n, HAMC_DATA_TYPE_t *C1, HAMC_DATA_TYPE_t *C2, 
    int ld) {
    int i;
    for (i = 0; i < n; i++) {
        HAMC_DATA_TYPE_t temp;
        temp = C1[i*ld];
        C1[i*ld] = C2[i*ld];
        C2[i*ld] = temp;
    }
}

/*
    // Call 3rd kernel: Update trailing matrix C ^= A & B, 
    // where A is A(k+1:n, k), B is A(k, k+1 : n), C is A(k+1: n, k+1:n)
    LU_GF2_update_trailing_cpu(n - k - 1,
        n - k - 1,
        &A->data[(k + 1) * n + k],
        &A->data[k * n + k + 1],
        &A->data[(k + 1) * n + (k + 1)]);
*/
void LU_GF2_update_trailing_cpu(int m , int n, HAMC_DATA_TYPE_t *A,
    HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C, int ld) {
    int i, j;
    for (i = 0; i < m; i++) { // row
        for (j = 0; j < n; j++) { // cols
            C[i * ld + j] ^= A[i * ld] & B[j];
        }
    }
}

bin_matrix inverse_GF2_cpu(bin_matrix A)
{
    bool verbose = false;

    int n = A->rows;

    // 
    int *IPIV = (int *)malloc(n*sizeof(int));

    /* LU decomposition */
    for (int k = 0; k < n; k++) {

        // Call 1st kernel: find the nearest ‘1’ on the subdiagonal column 
        // (column k, rows k:n), put the row number in IPIV[k]
        LU_GF2_find_max_cpu(n - k, &A->data[k * n + k], n, &IPIV[k], k);

        // Call 2nd kernel that swaps rows k and IPIV[k]

        LU_GF2_swap_rows_cpu(n, &A->data[k * n], &A->data[IPIV[k] * n]);

        // Call 3rd kernel: Update trailing matrix C ^= A & B, 
        // where A is A(k+1:n, k), B is A(k, k+1 : n), C is A(k+1: n, k+1:n)
        LU_GF2_update_trailing_cpu(n - k - 1,
            n - k - 1,
            &A->data[(k + 1) * n + k],
            &A->data[k * n + k + 1],
            &A->data[(k + 1) * n + (k + 1)],
            n);
    }

    if (verbose) {
        printf("\nA after LU decomposition (CPU):\n");
        print_bin_matrix(A);
        printf("\nIPIV:\n");
        for (int i = 0; i < A->rows; i++) {
            printf("%d ", IPIV[i]);
        }
        printf("\n");
    }



    /* Forward & Backward Substitution */
    bin_matrix IA = mat_init_cpu(n, n);

    //IA = I
    make_indentity_cpu(IA);


    // Forward
    for (int j = 0; j < n; j++) { // cols
        for (int i = j - 1; i >= 0; i--) { // rows from bottom to top
            IA->data[i*n + j] = A->data[i*n + j];
            for (int k = i+1; k < j; k++) {
                IA->data[i*n + j] ^= IA->data[k*n + j] & A->data[i*n + k];
            }
        }
    }

    // Backward
    for (int j = n - 1; j >= 0; j--) { // cols from right to left
        for (int i = 0; i < n; i++) { // rows from top to bottom
            //IA->data[i*n + j] = A->data[i*n + j];
            for (int k = j+1; k < n; k++) {
                IA->data[i*n + j] ^= IA->data[i*n + k] & A->data[k*n + j];
            }
        }
    }

    if (verbose) {
        printf("\nIA after backwards substition:\n");
        print_bin_matrix(IA);
    }

    for (int k = n - 1; k >= 0; k--) { // cols from right to left
        LU_GF2_swap_cols_cpu(n, &IA->data[k], &IA->data[IPIV[k]], n);
    }

    if (verbose) {
        printf("\nsolution:\n");
        print_bin_matrix(IA);
    }

    return IA;
}

#endif /* REFERENCE_CPU_C_H_N */