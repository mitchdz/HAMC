#include "MultiplyMatrix.h"

//Return the transpose of the matrix A
bin_matrix transpose(bin_matrix A)
{
  bin_matrix B;
  B = mat_init(A->cols, A->rows);
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      set_matrix_element(B, j, i, mat_element(A, i, j));
    }
  }
  return B;
}


//Multiplication of two matrices A and B stored in C
bin_matrix matrix_mult_cpu(bin_matrix A, bin_matrix B)
{
  if (A->cols != B->rows)
  {
    printf("Matrices are incompatible, check dimensions...\n");
    exit(0);
  }
  
  bin_matrix C;
  C = mat_init(A->rows, B->cols);
  bin_matrix B_temp = transpose(B);

  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0  ; j < B->cols; j++)
    {
      unsigned short val = 0;
      for(int k = 0; k < B->rows; k++)
      {
        val = (val ^ (mat_element(A, i, k) & mat_element(B_temp, j, k)));
      }
      mat_element(C, i, j) = val;
    }
  }
    
  return C;
}
