#ifndef TEST
#define TEST

#DEFINE ushort unsigned short

#define mat_element(mat, row_idx, col_idx) \
    mat->data[row_idx * (mat->cols) + col_idx]

typedef struct matrix
{
   int rows;             //number of rows.
   int cols;             //number of columns.
   ushort *data;
}*bin_matrix;

void* safe_malloc(size_t n);

bin_matrix mat_init(int rows, int cols);

ushort *dataGen(int rows, int cols);

#endif