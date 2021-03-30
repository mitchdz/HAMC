#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

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


/* function declarations */
bin_matrix mat_init_cpu(int rows, int cols);
void reset_row_cpu(unsigned short* row, int min, int max);
bin_matrix get_error_vector_cpu(int len, int t);
int random_val(int min, int max, unsigned seed);

void* safe_malloc(size_t n);
mdpc qc_mdpc_init_cpu(int n0, int p, int w, int t);


/* matrix function declarations */
bin_matrix generator_matrix_cpu(mdpc code);
void set_matrix_element_cpu(bin_matrix A, int row_idx, int col_idx, unsigned short val);
unsigned short get_matrix_element_cpu(bin_matrix mat, int row_idx, int col_idx);
bin_matrix parity_check_matrix_cpu(mdpc code);
bin_matrix concat_vertical_cpu(bin_matrix A, bin_matrix B);
bin_matrix concat_horizontal_cpu(bin_matrix A, bin_matrix B);
void set_matrix_row_cpu(bin_matrix A, int row, unsigned short* vec);
bin_matrix matrix_mult_cpu(bin_matrix A, bin_matrix B);

bin_matrix transpose_cpu(bin_matrix A);
void make_indentity_cpu(bin_matrix A);

bin_matrix circ_matrix_inverse_cpu(bin_matrix A);


//Set the value of matix element at position given by the indices to "val"
void set_matrix_element_cpu(bin_matrix A, int row_idx, int col_idx, unsigned short val)
{
  if(row_idx < 0 || row_idx >= A->rows || col_idx < 0 || col_idx >= A->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  mat_element(A, row_idx, col_idx) = val;
}


//Reset all positions in the row to 0
void reset_row_cpu(unsigned short* row, int min, int max)
{
    int i;
    for(i = min; i < max + 1; i++) {
        row[i] = 0;
    }
}
//Generate a random error vector of length len of weight t
bin_matrix get_error_vector_cpu(int len, int t)
{
    bin_matrix error = mat_init_cpu(1, len);
    int weight = 0;
    int idx;
    while(weight < t) {
        idx = random_val(1, len - 1, -1);
        if(!get_matrix_element_cpu(error, 0, idx)) {
            set_matrix_element_cpu(error, 0, idx, 1);
            weight++;
        }
    }
    return error;
}

//Initialize the mceliece cryptosystem
mcc mceliece_init_cpu(int n0, int p, int w, int t)
{
    mcc crypt;
    crypt = (mcc)safe_malloc(sizeof(struct mceliece));
    crypt->code = qc_mdpc_init_cpu(n0, p, w, t);
    crypt->public_key = generator_matrix_cpu(crypt->code);
    //printf("mceliece generated...\n");
    return crypt;
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
//Returns a random integer in the range [min, max]
int random_val(int min, int max, unsigned seed)
{
    int r;
    const unsigned int range = 1 + max - min;
    const unsigned int buckets = RAND_MAX / range;
    const unsigned int limit = buckets * range;

    do {
        r = rand();
    } while (r >= limit);

    return min + (r / buckets);
}


void* safe_malloc(size_t n)
{
    void* p = malloc(n);
    if (!p)
    {
        fprintf(stderr, "Out of memory(%lu bytes)\n",(size_t)n);
        exit(EXIT_FAILURE);
    }
    return p;
}
//Rotate the row x positions to the right
unsigned short* shift_cpu(unsigned short* row, int x, int len)
{
    unsigned short* temp = (unsigned short*)calloc(len, sizeof(unsigned short));
    int i;
    for(i = 0; i < len; i++)
    {
      temp[(i + x) % len] = row[i];
    }
    return temp;
}


//Set the indicated row of the matrix A equal to the vector vec
void set_matrix_row_cpu(bin_matrix A, int row, unsigned short* vec)
{
  if(row < 0 || row >= A->rows)
  {
    printf("Row index out of range\n");
    exit(0);
  }
  for(int i = 0; i < A->cols; i++)
  {
    set_matrix_element_cpu(A, row, i, vec[i]);
  }
}

//Create a binary circular matrix
bin_matrix make_matrix_cpu(int rows, int cols, unsigned short* vec, int x)
{
    bin_matrix mat = mat_init_cpu(rows, cols);
    set_matrix_row_cpu(mat, 0, vec);
    int i;
    for(i = 1; i < rows; i++)
    {
      vec = shift_cpu(vec, x, cols);
      set_matrix_row_cpu(mat, i, vec);
    }
    return mat;
}

//Splice the row for the given range (does not include max)
ushort* splice_cpu(ushort* row, int min, int max)
{
    ushort* temp = (ushort*)calloc(max - min, sizeof(ushort));
    int i;
    for(i = min; i < max; i++)
    {
      temp[i - min] = row[i];
    }
    return temp;
}

//Constructing the pariy check matrix
bin_matrix parity_check_matrix_cpu(mdpc code)
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    bin_matrix H = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, 0, code->p), 1);
    int i;
    for(i = 1; i < code->n0; i++)
    {
      bin_matrix M = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, i * code->p, (i + 1) * code->p), 1);
      H = concat_horizontal_cpu(H, M);
    }
    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("Time for H: %f\n", cpu_time_used);
    // printf("H: \n");
    // print_matrix(H);
    //printf("Parity matrix generated...\n");
    return H;
}

//Return the transpose of the matrix A
bin_matrix transpose_cpu(bin_matrix A)
{
  bin_matrix B;
  B = mat_init_cpu(A->cols, A->rows);
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      set_matrix_element_cpu(B, j, i, mat_element(A, i, j));
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
  C = mat_init_cpu(A->rows, B->cols);
  bin_matrix B_temp = transpose_cpu(B);

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


//Constructing the generator matrix
bin_matrix generator_matrix_cpu(mdpc code)
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    bin_matrix H = parity_check_matrix_cpu(code);


    //End of modified code
    printf("Construction of G started...\n");
    bin_matrix H_inv = circ_matrix_inverse_cpu(make_matrix_cpu(code->p, code->p, splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 1));
    //printf("H_inv generated...\n");
    //printf("stop\n");
    bin_matrix H_0 = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, 0, code->p), 1);
    bin_matrix Q = transpose_cpu(matrix_mult_cpu(H_inv,  H_0));
    //printf("Transpose obtained...\n");
    bin_matrix M;
    int i;
    for(i = 1; i < code->n0 - 1; i++) {
      M = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, i * code->p, (i + 1) * code->p), 1);
      M = transpose_cpu(matrix_mult_cpu(H_inv, M));
      Q = concat_vertical_cpu(Q, M);
    }
    bin_matrix I = mat_init_cpu(code->k, code->k);
    make_indentity_cpu(I);
    bin_matrix G = concat_horizontal_cpu(I, Q);

    //bin_matrix G = mat_kernel(H);
    //G = matrix_rref(G);
    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("Time for G: %f\n", cpu_time_used);
    printf("Generator matrix generated....\n");
    return G;
}

//Return the weight of the given row from the indices min to max
int get_row_weight(unsigned short* row, int min, int max)
{
    int weight = 0;
    int i;
    for(i = min; i < max + 1; i++) {
        if(row[i] == 1) {
            weight++;
        }
    }
    return weight;
}


mdpc qc_mdpc_init_cpu(int n0, int p, int w, int t)
{
    mdpc code;
    code = (mdpc)safe_malloc(sizeof(struct qc_mdpc));
    code->n0 = n0;
    code->p = p;
    code->w = w;
    code->t = t;
    code->n = n0 * p;
    code->r = p;
    code->k = (n0 - 1) * p;
    unsigned seed;
    code->row = (unsigned short*)calloc(n0 * p, sizeof(unsigned short));
    printf("Input seed or -1 to use default seed: ");
    scanf("%u", &seed);
    time_t tx;
    if(seed == -1) {
        srand((unsigned) time(&tx));
    } else {
        srand(seed);
    }

    while(1) {
        int flag = 0;
        int idx;
        while(flag < w)
        {
            idx = random_val(0, (n0 * p) - 1, seed);
            if(!code->row[idx])
            {
              code->row[idx] = 1;
              flag = flag + 1;
            }
        }
        if((get_row_weight(code->row, (n0 - 1) * p, (n0 * p)-1)) % 2 == 1)
        {
            break;
        }
        reset_row_cpu(code->row, 0, n0 * p);
    }
    printf("MDPC code generated....\n");
    return code;
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
bin_matrix circ_matrix_inverse_cpu(bin_matrix A)
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

void run_decryption_cpu(const char* outputFileName, int n, int p, int t, int w,
    int seed)
{

}


void run_encryption_cpu(const char* inputFileName, const char* outputFileName,
        int n, int p, int t, int w, int seed)
{
    cudaEvent_t astartEvent, astopEvent;

    int numARows, numAColumns;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);

    // retrieve message from file
    /* wbImport only reads and writes float, so we need to convert that */
    float *hostAFloats = (float *)wbImport(inputFileName, &numARows, &numAColumns);
    ushort *hostA = (ushort *)malloc(numARows*numAColumns * sizeof(ushort));
    for (int i = 0; i < numARows*numAColumns; i++)
        hostA[i] = (ushort)hostAFloats[i];

    // initialize encryption algo
    mcc crypt;
    crypt = (mcc)safe_malloc(sizeof(struct mceliece));
    crypt->code = qc_mdpc_init_cpu(n, p, w, t);
    crypt->public_key = generator_matrix_cpu(crypt->code);
    //printf("mceliece generated...\n");

    /* get error vector */

    /* add message, public key, and error */

    /* determine error length and weight */
    int error_length = 0;
    int error_weight = 0;

    ushort *error = get_error_vector_cpu(error_length, error_weight)->data;

}
