#ifndef HAMC_MATRIX_MULTIPLY
#define HAMC_MATRIX_MULTIPLY

#include "hamc_common.h"

bin_matrix matrix_mult_cpu(bin_matrix A, bin_matrix B);
bin_matrix transpose_cpu(bin_matrix A);

#endif /* HAMC_MATRIX_MULTIPLY */
