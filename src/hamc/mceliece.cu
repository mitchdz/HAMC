#ifndef HAMC_MCELIECE_H
#define HAMC_MCELIECE_H

#include "hamc_cpu_code.c"
#include "encrypt.cu"
#include "TransposeMatrix.cu"
#include "LU_inverse_plain.cu"


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
    clock_t start, end, 
            inverse_start, inverse_end,
            multiply_start, multiply_end,
            transpose_start, transpose_end,
            concat_start, concat_end;

    double cpu_time_used, 
           inverse_time_used,
           multiply_time_used,
           transpose_time_used,
           concat_time_used;


    start = clock();


    bin_matrix H = make_matrix_cpu(code->p,
                    code->p,
                    splice_cpu(code->row, (code->n0 - 1) * code->p, code->n),
                    1);
    printf("Construction of G started...\n");

    inverse_start = clock();

    //End of modified code
    //TODO: call GPU inverse
    inverse_start = clock();


    printf("Performing Inverse...\n");
    //bin_matrix H_inv = circ_matrix_inverse_cpu(H);
    bin_matrix H_inv = inverse_GF2_LU_gpu(H, false);


    inverse_end = clock();

    inverse_time_used = ((double) (inverse_end - inverse_start))/ CLOCKS_PER_SEC;
    printf("Inverse time used: %f\n", inverse_time_used);


    //printf("H_inv generated...\n");
    //printf("stop\n");
    bin_matrix H_0 = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, 0,
               code->p), 1);


    printf("Performing matrix multiplication..\n");
    multiply_start = clock();
    
    //bin_matrix H_inv_times_H0 = matrix_mult_cpu(H_inv, H_0);
    bin_matrix H_inv_times_H0 = run_mult_kernel(H_inv, H_0, 16);
    multiply_end = clock();
    multiply_time_used = ((double) (multiply_end - multiply_start))/ CLOCKS_PER_SEC;
    printf("Multiply time used: %f\n", multiply_time_used);


    transpose_start = clock();

    //TODO: transpose kernel
    // Tranpose H_inv*M
    //bin_matrix Q = transpose_cpu(H_inv_times_H0);
    bin_matrix Q = run_transpose_kernel(H_inv_times_H0);

    transpose_end = clock();
    transpose_time_used = ((double) (transpose_end - transpose_start))/ CLOCKS_PER_SEC;
    printf("Transpose time used: %f\n", transpose_time_used);


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


    concat_start = clock();

    bin_matrix G = concat_horizontal_cpu(I, Q);

    concat_end = clock();
    concat_time_used = ((double) (concat_end - concat_start))/ CLOCKS_PER_SEC;
    printf("concat time used: %f\n", concat_time_used);

    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("Time for G: %f\n", cpu_time_used);
    printf("Generator matrix generated....\n");

    printf("\tInverse:   %f - %.2f%%\n",
            inverse_time_used, 100*(inverse_time_used/cpu_time_used));

    printf("\tMultiply:  %f - %.2f%%\n",
            multiply_time_used, 100*(multiply_time_used/cpu_time_used));

    printf("\tTranspose: %f - %.2f%%\n",
            transpose_time_used, 100*(transpose_time_used/cpu_time_used));

    return G;

}

#endif /* HAMC_MCELIECE_H */
