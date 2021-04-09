#ifndef HAMC_MCELIECE_H
#define HAMC_MCELIECE_H

#include "hamc_cpu_code.c"
#include "encrypt.cu"

bin_matrix generator_matrix_gpu(mdpc code);

//Initialize the mceliece cryptosystem
mcc mceliece_init_gpu(int n0, int p, int w, int t, unsigned seed)
{
    mcc crypt;
    crypt = (mcc)safe_malloc(sizeof(struct mceliece));
    crypt->code = qc_mdpc_init_cpu(n0, p, w, t, seed);
    crypt->public_key = generator_matrix_gpu(crypt->code);
    //printf("mceliece generated...\n");
    return crypt;
}

bin_matrix generator_matrix_gpu(mdpc code)
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    //bin_matrix H = parity_check_matrix_cpu(code);

    //End of modified code
    printf("Construction of G started...\n");
    bin_matrix H_inv = circ_matrix_inverse_cpu(make_matrix_cpu(code->p, code->p,
               splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 1));

    //printf("H_inv generated...\n");
    //printf("stop\n");
    bin_matrix H_0 = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, 0,
               code->p), 1);

    //bin_matrix H_inv_times_H0 = mat_init_cpu(H_inv->rows, H_inv->cols);
    bin_matrix H_inv_times_H0 = run_matrix_multiply_kernel(H_inv, H_0);

    //TODO: transpose kernel
    // Tranpose H_inv*M
    //bin_matrix Q = transpose_cpu(H_inv_times_H0);
    bin_matrix Q = run_transpose_kernel(H_inv_times_H0);

    //printf("Transpose obtained...\n");
    bin_matrix M;

    int i;
    for(i = 1; i < code->n0 - 1; i++) {
        M = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, i * code->p, (i + 1) * code->p), 1);
        M = transpose_cpu(matrix_mult_cpu(H_inv, M));
        Q = concat_vertical_cpu(Q, M);
    }
    bin_matrix I = mat_init_cpu(code->k, code->k);
    make_indentity_cpu(I);
    bin_matrix G = concat_horizontal_cpu(I, Q);

    //bin_matrix G = mat_kernel(H);
    //G = matrix_rref(G);
    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("Time for G: %f\n", cpu_time_used);
    printf("Generator matrix generated....\n");
    return G;

}




#endif /* HAMC_MCELIECE_H */
