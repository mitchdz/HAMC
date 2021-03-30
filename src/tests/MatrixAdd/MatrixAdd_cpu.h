#ifndef ADD_MATRIX_CPU_H
#define ADD_MATRIX_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

#define ushort unsigned short

//FUNCTIONS go in here
void set_matrix_element(bin_matrix A, int row_idx, int col_idx, bin_matrix val);
bin_matrix MatrixAdd_cpu(bin_matrix *A, bin_matrix *B);


#ifdef __cplusplus
}
#endif

#endif //ADD_MATRIX_CPU_H
