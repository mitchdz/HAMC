#include <stdio.h>
#include <stdlib.h>
#include "MatrixAdd_cpu.h"

#define HAMC_DATA_TYPE_t HAMC_DATA_TYPE_t

#define mat_element(mat, cols, row_idx, col_idx) \
  mat[(row_idx * cols) + col_idx]
  
//initialize the matrix
HAMC_DATA_TYPE_t mat_init(int rows, int cols)
{
  if(rows <= 0 || cols <= 0)
  {
    return NULL;
  }
  HAMC_DATA_TYPE_t A;
  A = (HAMC_DATA_TYPE_t)safe_malloc(sizeof(struct matrix));
  A->cols = cols;
  A->rows = rows; 
  A->data = (HAMC_DATA_TYPE_t *)safe_malloc(rows*cols*sizeof(HAMC_DATA_TYPE_t)); 
  return A;
}

//Set the value of matix element at position given by the indices to "val"
void set_matrix_element(HAMC_DATA_TYPE_t A, int row_idx, int col_idx, HAMC_DATA_TYPE_t val)
{
  if(row_idx < 0 || row_idx >= A->rows || col_idx < 0 || col_idx >= A->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  mat_element(A, row_idx, col_idx) = val;
}

HAMC_DATA_TYPE_t matrix_add(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B)
{
    if(A->rows != B->rows || A->cols != B->cols)
    {
        printf("Incompatible dimensions for matrix addition. \n");
        exit(0);
    }
    HAMC_DATA_TYPE_t temp mat_init(A->rows, A->cols);
    for(int i = 0; i < A->rows; i++)
    {
        for(int j = 0; j < A->cols; j++)
        {
            set_matrix_element(temp, i, j, mat_element(A, i ,j) ^ mat_element(B, i ,j)));
        }
    }
    return temp;
}
