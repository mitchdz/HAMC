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

bool is_identity(ushort A)
{
  bool flag = true;
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      if(i == j)
      {
        if(mat_element(A, i, j) == 0)
        {
          flag = false;
          return flag;
        }
      }
      else
      {
        if(mat_element(A, i, j) == 1)
        {
          flag = false;
          return flag;
        }
      }
    }
  }
  return flag;
}

//Set matrix as identity matrix
void make_indentity(ushort A)
{
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      if(i == j)
      {
        mat_element(A, i, j) = 1;
      }
      else
      {
        mat_element(A, i, j) = 0;
      }
    }
  }
}

//Add row1 to row2 of matrix A
ushort add_rows(unshort A,int row1, int row2)
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

//Add the elements of row1 to row2 in the column index range [a,b]  
ushort add_rows_new(unshort A,int row1, int row2, int a, int b)
{
  if(row1 < 0 || row1 >= A->rows || row2 < 0 || row2 >= A->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  for(int i = a; i < b; i++)
  {
    mat_element(A, row2, i) = (mat_element(A, row1, i) ^ mat_element(A, row2, i));
  }
  return A;
}

//Inverse of matrix
ushort circ_matrix_inverse(unshort A)
{
  if(A->rows != A->cols)
  {
    printf("Inverse not possible...\n");
    exit(0);
  }

  if(is_identity(A))
  {
    return A;
  }

  unshort B;
  B = mat_init(A->rows, A->cols);
  make_indentity(B);


  int i;
  int flag, prev_flag = 0;

  for(i = 0; i < A->cols; i++)
  {
    if(mat_element(A, i, i) == 1)
    {      
      for(int j = 0; j <  A->rows; j++)
      {
        if(i != j && mat_element(A, j, i) == 1)
        {
          add_rows_new(B, i, j, 0, A->cols);
          add_rows_new(A, i, j, i, A->cols);
        }
      }
    }
    else
    {
      int k;
      for(k = i + 1; k < A->rows; k++)
      {
        if(mat_element(A, k, i) == 1)
        {
          add_rows(B, k, i);
          add_rows(A, k, i);
          i = i - 1;
          break;
        } 
      }
    }
  }
  //printf("Out of for loop...\n");
  if(!is_identity(A))
  {
    printf("Could not find inverse, exiting...\n");  
    exit(-1);
  }

  
  return B;
}

