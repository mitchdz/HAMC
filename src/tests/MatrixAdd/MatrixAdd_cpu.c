#include <stdio.h>
#include <stdlib.h>
#include "MatrixAdd_cpu.h"

#define ushort unsigned short

#define mat_element(mat, cols, row_idx, col_idx) \
  mat[(row_idx * cols) + col_idx]
  
//Set the value of matix element at position given by the indices to "val"
void set_matrix_element(bin_matrix A, int row_idx, int col_idx, bin_matrix val)
{
  if(row_idx < 0 || row_idx >= A->rows || col_idx < 0 || col_idx >= A->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  mat_element(A, row_idx, col_idx) = val;
}

bin_matrix MatrixAdd_cpu(bin_matrix *A, bin_matrix *B)
{
  //  if(A->rows != B->rows || A->cols != B->cols)
  //   {
  //       printf("Incompatible dimensions for matrix addition. \n");
  //      exit(0);
  //  }
  // bin_matrix temp mat_init(A->rows, A->cols);
    for(int i = 0; i < A->rows; i++)
    {
        for(int j = 0; j < A->cols; j++)
        {
            set_matrix_element(temp, i, j, mat_element(A, i ,j) ^ mat_element(B, i ,j)));
        }
    }
    return temp;
}
