#include "mceliece.h"
#include "qc_mdpc.h"
#include "matrix.h"


//Generate a random error vector of length len of weight t
bin_matrix get_error_vector_cpu(int len, int t)
{
    bin_matrix error = mat_init_cpu(1, len);
    int weight = 0;
    int idx;
    while(weight < t) {
        idx = random_val(1, len - 1, -1);
        if(!get_matrix_element_cpu(error, 0, idx)) {
            set_matrix_element_cpu(error, 0, idx, 1);
            weight++;
        }
    }
    return error;
}

//Initialize the mceliece cryptosystem
mcc mceliece_init_cpu(int n0, int p, int w, int t)
{
    mcc crypt;
    crypt = (mcc)safe_malloc(sizeof(struct mceliece));
    crypt->code = qc_mdpc_init_cpu(n0, p, w, t);
    crypt->public_key = generator_matrix_cpu(crypt->code);
    //printf("mceliece generated...\n");
    return crypt;
}
