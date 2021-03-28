#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../test.h"
#include "../test.cpp"

//Function to swap two rows of matrix A
void swap(bin_matrix A, int row1, int row2)
{
  if(row1 < 0 || row1 >= A->rows || row2 < 0 || row2 >= A->rows)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  int temp;
  for(int i = 0; i < A->cols; i++)
  {
    temp = mat_element(A, row1, i);
    mat_element(A, row1, i) = mat_element(A, row2, i);
    mat_element(A, row2, i) = temp;
  }
}

//Add row1 to row2 of matrix A
bin_matrix add_rows(bin_matrix A,int row1, int row2)
{
  if(row1 < 0 || row1 >= A->rows || row2 < 0 || row2 >= A->rows)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  for(int i = 0; i < A->cols; i++)
  {
    mat_element(A, row2, i) = (mat_element(A, row1, i) ^ mat_element(A, row2, i));
  }
  return A;
}

//Copy the data of matrix A to matrix B
bin_matrix mat_copy(bin_matrix A)
{
  bin_matrix B;
  int i;

  B = mat_init(A->rows, A->cols);
  memcpy(B->data, A->data, (A->rows)*(A->cols)*(sizeof(unsigned short)));
  return B;
}


//Function to obtain the row reduced echlon form of a matrix A
bin_matrix matrix_rref(bin_matrix A)
{
  int lead = 0;
  int row_count = A->rows;
  int col_count = A->cols;
  bin_matrix temp = mat_init(row_count, col_count);
  temp = mat_copy(A);

  int r = 0;
  while(r < row_count)
  {
    if(mat_element(temp, r, r) == 0)
    {
      int i;
      for(i = r + 1; i < temp->rows; i++)
      {
        printf("i: %dm element: %d\n", i, mat_element(temp, i, r));
        if(mat_element(temp, i, r) == 1)
        {
          swap(temp, r, i);
          break;
        }
      }
      if(i == row_count)
      {
      	printf("Matix cannot be transformed into row echlon form...");
        exit(1);
      }
    }
    else
    {
      for(int i = 0; i < row_count; i++)
      {
        if(mat_element(temp, i, r) == 1 && i != r)
        {
          add_rows(temp, r, i);
        }
      }
      r++;
    }
  }
  return temp;
}





//#define mat_element(mat, cols, row_idx, col_idx) \
//  mat[(row_idx * cols) + col_idx]
//
//
//void swap(ushort *A, int row1, int row2, int rows, int cols)
//{
//    int temp;
//    for(int i = 0; i < cols; i++)
//    {
//        temp = mat_element(A, cols, row1, i);
//        mat_element(A, cols, row1, i) = mat_element(A, cols, row2, i);
//        mat_element(A, cols, row2, i) = temp;
//    }
//}
//
//
//
//void add_rows(ushort *A, int row1, int row2, int rows, int cols)
//{
//  for(int i = 0; i < cols; i++)
//  {
//    mat_element(A, cols, row2, i) = \
//        (mat_element(A, cols, row1, i) ^ mat_element(A, cols, row2, i));
//  }
//}
//
//
//
////Function to obtain the row reduced echlon form of a matrix A
//void matrix_rref(ushort *A, ushort *B, int rows, int cols)
//{
//  ushort *temp = (ushort *)calloc(sizeof(ushort), rows * cols);
//  for (int i = 0; i < rows*cols;i++)
//      temp[i] = A[i];
//
//
//  int r = 0;
//  while(r < rows) {
//    if(mat_element(temp, cols, r, r) == 0) {
//      int i;
//      for(i = r + 1; i < rows; i++) {
//        if(mat_element(temp, cols, i, r) == 1) {
//          swap(temp, r, i, rows, cols);
//          break;
//        }
//      }
//      if(i == rows) {
//          printf("Matix cannot be transformed into row echlon form...");
//          exit(1);
//      }
//    }
//    else {
//      for(int i = 0; i < rows; i++) {
//        if(mat_element(temp, cols, i, r) == 1 && i != r) {
//          add_rows(temp, r, i, rows, cols);
//        }
//      }
//      r++;
//    }
//  }
//  return;
//}
