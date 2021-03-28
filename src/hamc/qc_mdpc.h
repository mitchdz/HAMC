#ifndef HAMC_QC_MDPC_H
#define HAMC_QC_MDPC_H

#include "hamc_common.h"

unsigned short* shift_cpu(unsigned short* row, int x, int len);
bin_matrix make_matrix_cpu(int rows, int cols, unsigned short* vec, int x);
ushort* splice_cpu(ushort* row, int min, int max);
bin_matrix parity_check_matrix_cpu(mdpc code);
bin_matrix generator_matrix_cpu(mdpc code);
int get_row_weight(unsigned short* row, int min, int max);
mdpc qc_mdpc_init_cpu(int n0, int p, int w, int t);

#endif /* HAMC_QC_MDPC_H */
