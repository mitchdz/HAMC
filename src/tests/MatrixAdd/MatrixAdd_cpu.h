#ifndef ADD_MATRIX_CPU_H
#define ADD_MATRIX_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

#define ushort unsigned short

//FUNCTIONS go in here
ushort mat_init(int rows, int cols);
void set_matrix_element(ushort A, int row_idx, int col_idx, ushort val);
ushort add_matrix(ushort *A, ushort *B);


#ifdef __cplusplus
}
#endif

#endif //ADD_MATRIX_CPU_H
