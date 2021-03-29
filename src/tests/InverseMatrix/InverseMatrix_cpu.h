#ifndef INVERSE_MATRIX_CPU_H
#define INVERSE_MATRIX_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

#define ushort unsigned short

ushort mat_init(int rows, int cols);
bool is_identity(ushort A);
void make_indentity(ushort A);
ushort add_rows(ushort A,int row1, int row2);
ushort add_rows_new(ushort A,int row1, int row2, int a, int b);
ushort circ_matrix_inverse(unshort A);



#ifdef __cplusplus
}
#endif

#endif //INVERSE_MATRIX_CPU_H
