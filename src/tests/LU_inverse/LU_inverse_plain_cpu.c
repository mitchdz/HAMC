#ifndef REFERENCE_CPU_C_H_N
#define REFERENCE_CPU_C_H_N

#include "../hamc/hamc_common.h"
#include "../hamc/hamc_cpu_code.c"


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

bin_matrix inverse_GF2_cpu(bin_matrix A, bool verbose)
{

    int n = A->rows;

    clock_t LU_start = clock();

    // 
    int *IPIV = (int *)malloc(n*sizeof(int));



    clock_t LU_decompose_start = clock();


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

    clock_t LU_decompose_end = clock();
    double LU_decompose_time = ((double) (LU_decompose_end - LU_decompose_start))/ CLOCKS_PER_SEC;

    if (verbose) {
        if (A->rows < 60) { 
            printf("\nA after LU decomposition (CPU):\n");
            print_bin_matrix(A);
            printf("\nIPIV (CPU):\n");
            for (int i = 0; i < A->rows; i++) {
                printf("%d ", IPIV[i]);
            }
            printf("\n");
        }
    }


    /* Forward & Backward Substitution */
    bin_matrix IA = mat_init_cpu(n, n);

    //IA = I
    make_indentity_cpu(IA);


    clock_t LU_forward_start = clock();

    // Forward
    for (int j = 0; j < n; j++) { // cols
        for (int i = j - 1; i >= 0; i--) { // rows from bottom to top
            IA->data[i*n + j] = A->data[i*n + j];
            for (int k = i+1; k < j; k++) {
                IA->data[i*n + j] ^= IA->data[k*n + j] & A->data[i*n + k];
            }
        }
    }
    clock_t LU_forward_end = clock();
    double LU_forward_time = ((double) (LU_forward_end - LU_forward_start))/ CLOCKS_PER_SEC;

    clock_t LU_backward_start = clock();

    // Backward
    for (int j = n - 1; j >= 0; j--) { // cols from right to left
        for (int i = 0; i < n; i++) { // rows from top to bottom
            //IA->data[i*n + j] = A->data[i*n + j];
            for (int k = j+1; k < n; k++) {
                IA->data[i*n + j] ^= IA->data[i*n + k] & A->data[k*n + j];
            }
        }
    }
    clock_t LU_backward_end = clock();
    double LU_backward_time = ((double) (LU_backward_end - LU_backward_start))/ CLOCKS_PER_SEC;

    if (verbose) {
        if (IA->rows < 60) {
            printf("\nIA after backwards substition (CPU):\n");
            print_bin_matrix(IA);
        }
    }

    clock_t LU_swap_start = clock();


    for (int k = n - 1; k >= 0; k--) { // cols from right to left
        LU_GF2_swap_cols_cpu(n, &IA->data[k], &IA->data[IPIV[k]], n);
    }

    clock_t LU_swap_end = clock();
    double LU_swap_time = ((double) (LU_swap_end - LU_swap_start))/ CLOCKS_PER_SEC;


    clock_t LU_end = clock();
    double LU_time = ((double) (LU_end - LU_start))/ CLOCKS_PER_SEC;


    if (verbose) {
        if (IA->rows < 60) {
            printf("\nsolution (CPU):\n");
            print_bin_matrix(IA);
        }
    }


    if (verbose) {
        printf("Total time for CPU LU Inverse: %.4lf\n", LU_time);
        printf("\tLU decomposition:       %.4lf\n", LU_decompose_time);
        printf("\tForward substitution:   %.4lf\n", LU_forward_time);
        printf("\tBackward substitution:  %.4lf\n", LU_backward_time);
        printf("\tFinal Swap:             %.4lf\n", LU_swap_time);
    }





    return IA;
}

#endif /* REFERENCE_CPU_C_H_N */