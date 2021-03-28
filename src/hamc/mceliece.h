#ifndef HAMC_MCELIECE_H
#define HAMC_MCELIECE_H

#include "hamc_common.h"

mcc mceliece_init_cpu(int n0, int p, int w, int t);
bin_matrix get_error_vector_cpu(int len, int t);

#endif /* HAMC_MCELIECE_H */
