#ifndef HAMC_MATRIX_H
#define HAMC_MATRIX_H

#include "hamc_common.h"

void set_matrix_element_cpu(bin_matrix A, int row_idx, int col_idx, ushort val);
void set_matrix_row_cpu(bin_matrix A, int row, ushort* vec);
unsigned short get_matrix_element_cpu(bin_matrix mat, int row_idx, int col_idx);
bin_matrix circ_matrix_inverse_cpu(bin_matrix A);
bool is_identity_cpu(bin_matrix A);
bin_matrix add_rows_new_cpu(bin_matrix A,int row1, int row2, int a, int b);
bin_matrix mat_init_cpu(int rows, int cols);
bin_matrix add_rows_cpu(bin_matrix A,int row1, int row2);
void make_indentity_cpu(bin_matrix A);
bin_matrix concat_horizontal_cpu(bin_matrix A, bin_matrix B);
bin_matrix concat_vertical_cpu(bin_matrix A, bin_matrix B);


#endif /*HAMC_MATRIX_H */
