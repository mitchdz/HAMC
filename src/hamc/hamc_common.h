#ifndef HAMC_COMMON_H
#define HAMC_COMMON_H

#include <stdlib.h>
#include <stdio.h>


typedef struct matrix
{
    int rows;             //number of rows.
    int cols;             //number of columns.
    unsigned short *data;
}*bin_matrix;

typedef struct qc_mdpc
{
    unsigned short* row;
    int n0, p, w, t, n, k, r;
}*mdpc;


typedef struct mceliece
{
    mdpc code;
    bin_matrix public_key;
}*mcc;


#define mat_element(mat, row_idx, col_idx) \
      mat->data[row_idx * (mat->cols) + col_idx]

#define ushort unsigned short


void* safe_malloc(size_t n);
int random_val(int min, int max, unsigned seed);
void reset_row_cpu(unsigned short* row, int min, int max);





#endif /* HAMC_COMMON_H */
