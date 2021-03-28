#include <stdio.h>
#include <stdlib.h>
#include "MatrixAdd_cpu.h"

#define ushort unsigned short

#define mat_element(mat, cols, row_idx, col_idx) \
  mat[(row_idx * cols) + col_idx]
  
//initialize the matrix
ushort mat_init(int rows, int cols)
{
  if(rows <= 0 || cols <= 0)
  {
    return NULL;
  }
  ushort A;
  A = (ushort)safe_malloc(sizeof(struct matrix));
  A->cols = cols;
  A->rows = rows; 
  A->data = (ushort *)safe_malloc(rows*cols*sizeof(ushort)); 
  return A;
}

//Set the value of matix element at position given by the indices to "val"
void set_matrix_element(ushort A, int row_idx, int col_idx, ushort val)
{
  if(row_idx < 0 || row_idx >= A->rows || col_idx < 0 || col_idx >= A->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  mat_element(A, row_idx, col_idx) = val;
}

ushort matrix_add(ushort *A, ushort *B)
{
    if(A->rows != B->rows || A->cols != B->cols)
    {
        printf("Incompatible dimensions for matrix addition. \n");
        exit(0);
    }
    ushort temp mat_init(A->rows, A->cols);
    for(int i = 0; i < A->rows; i++)
    {
        for(int j = 0; j < A->cols; j++)
        {
            set_matrix_element(temp, i, j, mat_element(A, i ,j) ^ mat_element(B, i ,j)));
        }
    }
    return temp;
}
