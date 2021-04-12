#ifndef HAMC_E2E_H
#define HAMC_E2E_H

#include <stdio.h>
#include <stdbool.h>
#include "hamc_cpu_code.c"
#include "mceliece.cu"
#include "decrypt.cu"


void test_gpu_e2e(int n0, int p, int t, int w, int seed, bool verbose)
{

    printf("GPU based mceliece cryptosystem test\n");

    printf("Starting Encryption...\n");
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    clock_t error_vector_start, error_vector_end;
    double error_vector_time_used;

    clock_t mceliece_init_start, mceliece_init_end;
    double mceliece_init_time_used;

    clock_t encrypt_start, encrypt_end;
    double encrypt_time_used;

    clock_t decrypt_start, decrypt_end;
    double decrypt_time_used;


    mceliece_init_start = clock();

    mcc crypt = mceliece_init_gpu(n0, p, w, t, seed);

    mceliece_init_end = clock();
    mceliece_init_time_used = ((double) (mceliece_init_end - mceliece_init_start))
        / CLOCKS_PER_SEC;


    bin_matrix msg = mat_init_cpu(1, crypt->code->k);
    //Initializing the message a random message
    for(int i = 0; i < crypt->code->k; i++)
    {
            int z = rand() % 2;
            set_matrix_element_cpu(msg, 0, i, z);
    }

    if (verbose) {
        printf("message:\n");
        for (int i = 0; i < msg->cols; i++)
            printf("%hu", msg->data[i]);
        printf("\n");

        printf("public key:\n");
        for (int i = 0; i < crypt->public_key->cols; i++)
            printf("%hu", crypt->public_key->data[i]);
        printf("\n");
    }


    error_vector_start = clock();

    bin_matrix error = get_error_vector_cpu(crypt->code->n, crypt->code->t);

    error_vector_end = clock();
    error_vector_time_used = ((double) (error_vector_end - error_vector_start))
        / CLOCKS_PER_SEC;


    if (verbose) {
        /* display error vector */
        printf("error vector:\n");
        for (int i = 0; i < error->cols; i++)
            printf("%hu", error->data[i]);
        printf("\n");
    }


    encrypt_start = clock();

    bin_matrix v = encrypt_cpu(msg, crypt);

    encrypt_end = clock();
    encrypt_time_used = ((double) (encrypt_end - encrypt_start))
        / CLOCKS_PER_SEC;


    if (verbose) {
        /* display encrypted data */
        printf("encrypted data (message * public key + error):\n");
        for (int i = 0; i < v->cols; i++)
            printf("%hu", v->data[i]);
        printf("\n");
    }


    decrypt_start = clock();
    //bin_matrix s = decrypt_cpu(v, crypt);
    bin_matrix s = decrypt_gpu(v, crypt);

    decrypt_end = clock();
    decrypt_time_used = ((double) (decrypt_end - decrypt_start))
        / CLOCKS_PER_SEC;


    if (verbose) {
        printf("decrypted data:\n");
        for (int i = 0; i < s->cols; i++)
            printf("%hu", s->data[i]);
        printf("\n");
    }

    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    if(mat_is_equal_cpu(msg, s)) {
            printf("Decryption successful...\n");
            printf("Time taken by cryptosystem: %f\n", cpu_time_used);
    } else {
            printf("Failure....\n");
    }
    delete_mceliece_cpu(crypt);

    printf("Time taken by individual phases:\n");
    printf("\tMcEliece init: %f - %.2f\%\n", mceliece_init_time_used, 100* (mceliece_init_time_used/cpu_time_used));
    //printf("\terror vector generation time: %f - %.2f\%\n",
    //        error_vector_time_used, 100* (error_vector_time_used/cpu_time_used));
    printf("\tencryption:    %f - %.2f\%\n", encrypt_time_used, 100*(encrypt_time_used/cpu_time_used));
    printf("\tdecryption:    %f - %.2f\%\n", decrypt_time_used, 100*(decrypt_time_used/cpu_time_used));

    free(v);
    free(s);
    free(msg);

    return;
}


#endif /* HAMC_E2E_H */
