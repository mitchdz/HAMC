#include "matrix.h"


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


//Concatenate the matrices A and B vertically
bin_matrix concat_vertical_cpu(bin_matrix A, bin_matrix B)
{
  if(A->cols != B->cols)
  {
    printf("Incompatible dimensions of the two matrices. Number of rows should be same.\n");
    exit(0);
  }
  bin_matrix temp = mat_init_cpu(A->rows + B->rows, A->cols);
  for(int i = 0; i < temp->rows; i++)
  {
    for(int j = 0; j < temp->cols; j++)
    {
      if(i < A->rows)
      {
        set_matrix_element_cpu(temp, i, j, mat_element(A, i, j));
      }
      else
      {
        set_matrix_element_cpu(temp, i, j, mat_element(B, i - A->rows, j));
      }
    }
  }
  return temp;
}


//Concatenate the matrices A and B as [A|B]
bin_matrix concat_horizontal_cpu(bin_matrix A, bin_matrix B)
{
    if(A->rows != B->rows) {
        printf("Incompatible dimensions of the two matrices. Number of rows should be same.\n");
        exit(0);
    }
    bin_matrix temp = mat_init_cpu(A->rows, A->cols + B->cols);
    for(int i = 0; i < temp->rows; i++) {
        for(int j = 0; j < temp->cols; j++) {
            if(j < A->cols) {
                set_matrix_element_cpu(temp, i, j, mat_element(A, i, j));
            }
            else {
                set_matrix_element_cpu(temp, i, j, mat_element(B, i, j - A->cols));
            }
        }
    }
    return temp;
}

//Set matrix as identity matrix
void make_indentity_cpu(bin_matrix A)
{
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
          if(i == j) {
              mat_element(A, i, j) = 1;
          }
          else {
              mat_element(A, i, j) = 0;
          }
        }
    }
}

//Add row1 to row2 of matrix A
bin_matrix add_rows_cpu(bin_matrix A,int row1, int row2)
{
    if(row1 < 0 || row1 >= A->rows || row2 < 0 || row2 >= A->rows) {
        printf("Matrix index out of range\n");
        exit(0);
    }
    for(int i = 0; i < A->cols; i++) {
        mat_element(A, row2, i) = (mat_element(A, row1, i) ^ mat_element(A, row2, i));
    }
    return A;
}


//initialize the matrix
bin_matrix mat_init_cpu(int rows, int cols)
{
    if(rows <= 0 || cols <= 0) {
      return NULL;
    }
    bin_matrix A;
    A = (bin_matrix)safe_malloc(sizeof(struct matrix));
    A->cols = cols;
    A->rows = rows;
    A->data = (unsigned short *)safe_malloc(rows*cols*sizeof(unsigned short));
    return A;
}

//Add the elements of row1 to row2 in the column index range [a,b]
bin_matrix add_rows_new_cpu(bin_matrix A,int row1, int row2, int a, int b)
{
    if(row1 < 0 || row1 >= A->rows || row2 < 0 || row2 >= A->cols) {
      printf("Matrix index out of range\n");
      exit(0);
    }
    for(int i = a; i < b; i++) {
      mat_element(A, row2, i) = (mat_element(A, row1, i) ^ mat_element(A, row2, i));
    }
    return A;
}


bool is_identity_cpu(bin_matrix A)
{
    bool flag = true;
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            if(i == j) {
                if(mat_element(A, i, j) == 0) {
                    flag = false;
                    return flag;
                }
            }
            else {
                if(mat_element(A, i, j) == 1) {
                    flag = false;
                    return flag;
                }
            }
        }
    }
    return flag;
}


//Inverse of matrix
bin_matrix circ_matrix_inverse(bin_matrix A)
{
    if(A->rows != A->cols) {
      printf("Inverse not possible...\n");
      exit(0);
    }

    if(is_identity_cpu(A)) {
      return A;
    }

    bin_matrix B;
    B = mat_init_cpu(A->rows, A->cols);
    make_indentity_cpu(B);

    int i;
    int flag, prev_flag = 0;

    for(i = 0; i < A->cols; i++) {
        if(mat_element(A, i, i) == 1) {
            for(int j = 0; j <  A->rows; j++) {
                if(i != j && mat_element(A, j, i) == 1) {
                    add_rows_new_cpu(B, i, j, 0, A->cols);
                    add_rows_new_cpu(A, i, j, i, A->cols);
                }
            }
        }
        else{
            int k;
            for(k = i + 1; k < A->rows; k++) {
                if(mat_element(A, k, i) == 1) {
                    add_rows_cpu(B, k, i);
                    add_rows_cpu(A, k, i);
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
