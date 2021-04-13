#ifndef DEBUG_TEST_INVERSE_C
#define DEBUG_TEST_INVERSE_C

#include <stdio.h>
#include "../hamc/hamc_cpu_code.c"


//Inverse of matrix
bin_matrix my_circ_matrix_inverse_cpu(bin_matrix A)
{
    if(A->rows != A->cols) {
      printf("Inverse not possible...\n");
      exit(0);
    }

    if(is_identity_cpu(A)) {
      printf("already identity\n");
      return A;
    }

    bin_matrix B;
    B = mat_init_cpu(A->rows, A->cols);
    make_indentity_cpu(B);

    int i;

    for(i = 0; i < A->cols; i++) {
        if(mat_element(A, i, i) == 1) {
            for(int j = 0; j <  A->rows; j++) {
                if(i != j && mat_element(A, j, i) == 1) {
                    add_rows_new_cpu(B, i, j, 0, A->cols);
                    add_rows_new_cpu(A, i, j, i, A->cols);
                    //printf("i: %dj: %d\n", i, j);
                    //printf("A:\n");
                    //print_bin_matrix(A);
                    //printf("B:\n");
                    //print_bin_matrix(B);
                }
            }
        }
        else{
            int k;
            for(k = i + 1; k < A->rows; k++) {
                if(mat_element(A, k, i) == 1) {
                    add_rows_cpu(B, k, i);
                    add_rows_cpu(A, k, i);
                    //printf("i: %dk: %d\n", i, k);
                    //printf("A:\n");
                    //print_bin_matrix(A);
                    //printf("B:\n");
                    //print_bin_matrix(B);
                    i = i - 1;
                    break;
                }
            }
        }
    }
    //printf("Out of for loop...\n");
    if(!is_identity_cpu(A))
    {
      printf("Could not find inverse, exiting...\n");
      exit(-1);
    }
    return B;
}

#endif /* DEBUG_TEST_INVERSE_C */
