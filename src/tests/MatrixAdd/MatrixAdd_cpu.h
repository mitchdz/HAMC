#ifndef ADD_MATRIX_CPU_H
#define ADD_MATRIX_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../../hamc/hamc_common.h"


//FUNCTIONS go in here
HAMC_DATA_TYPE_t mat_init(int rows, int cols);
void set_matrix_element(HAMC_DATA_TYPE_t A, int row_idx, int col_idx, HAMC_DATA_TYPE_t val);
HAMC_DATA_TYPE_t add_matrix(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B);


#ifdef __cplusplus
}
#endif

#endif //ADD_MATRIX_CPU_H
