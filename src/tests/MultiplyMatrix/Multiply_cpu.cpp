#define ushort unsigned short

#define mat_element(mat, row_idx, col_idx) \
    mat->data[row_idx * (mat->cols) + col_idx]

typedef struct matrix
{
   int rows;             //number of rows.
   int cols;             //number of columns.
   ushort *data;
}*bin_matrix;

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

bin_matrix mat_init(int rows, int cols){
	if(rows <= 0 || cols <= 0)
		return NULL;

	bin_matrix A;
	A = (bin_matrix)safe_malloc(sizeof(struct matrix));
	A->cols = cols;
	A->rows = rows; 
	A->data = (ushort *)safe_malloc(rows*cols*sizeof(ushort)); 
	return A;
}

bin_matrix transpose(bin_matrix A){
    bin_matrix B;
    B = mat_init(A->cols, A->rows);
    for(int i = 0; i < A->rows; i++){
        for(int j = 0; j < A->cols; j++){
            set_matrix_element(B, j, i, mat_element(A, i, j));
        }
    }
    return B;    
}

bin_matrix mult_matrix(bin_matrix A, bin_matrix B)
{
    if (A->cols != B->rows){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }

    bin_matrix C;
    C = mat_init(A->rows, B->cols);
    bin_matrix B_temp = transpose(B);

    for(int i = 0; i < A->rows; i++){
        for(int j = 0; j < B->cols; j++){
            unsigned short val = 0;
            for(int k = 0; k < B->rows; k++){
                val = (val ^ (mat_element(A, i, k) & mat_element(B_temp, j, k)));
            }
            mat_element(C, i, j) = val;
        }
    }
    return C;
}