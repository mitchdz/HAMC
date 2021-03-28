#include <stdio.h>

#include "keygen.h"
#include "hamc_common.h"
#include "qc_mdpc.h"
#include "mceliece.h"

void run_keygen_gpu(const char* outputFileName, int n, int p, int t, int w,
    int seed)
{
    //TODO: implement
}

//Return the matrix element at position given by the indices
unsigned short get_matrix_element_cpu(bin_matrix mat, int row_idx, int col_idx)
{
  if(row_idx < 0 || row_idx >= mat->rows || col_idx < 0 || col_idx >= mat->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  return mat->data[row_idx * (mat->cols) + col_idx];
}


void run_keygen_cpu(const char* outputFileName, int n, int p, int t, int w,
    int seed)
{
    mcc crypt = mceliece_init_cpu(n, p, w, t);

    bin_matrix H = parity_check_matrix_cpu(crypt->code);
    bin_matrix G = generator_matrix_cpu(crypt->code);
    FILE *fp1, *fp2;
    fp1 = fopen("Private_Key.txt", "a");
    fprintf(fp1, "Private Key: Parity Check Matrix: \n");
    for(int i = 0; i < H->rows; i++) {
        for(int j = 0; j < H->cols; j++) {
            fprintf(fp1, "%hu ", get_matrix_element_cpu(H, i, j));
        }
        fprintf(fp1, "\n \n");
    }
    fclose(fp1);

    fp2 = fopen("Public_Key.txt", "a");
    fprintf(fp2, "Public Key: Generator Matrix: \n");
    for(int i = 0; i < G->rows; i++) {
        for(int j = 0; j < G->cols; j++) {
            fprintf(fp2, "%hu ", get_matrix_element_cpu(G, i, j));
        }
        fprintf(fp2, "\n \n");
    }
    fclose(fp2);
    printf("Keys Generated...\n");

}
