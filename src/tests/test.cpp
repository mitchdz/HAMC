#include "test.h"

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

//initialize the matrix
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

//data generation
ushort *dataGen(int rows, int cols){
    ushort *data = (ushort *)malloc(sizeof(ushort) * cols * rows);
    for(int i = 0; i < rows * cols; i++){
        data[i] = (ushort)(rand() % 2);
    }
    return data;
}
