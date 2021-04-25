#ifndef HAMC_E2E_H
#define HAMC_E2E_H

#include <stdio.h>
#include <stdbool.h>
#include "hamc_cpu_code.c"
#include "mceliece.cu"
#include "decrypt.cu"
#include "hamc_common.h"



void test_hamc_e2e(int n0, int p, int t, int w, int seed, bool cpu, bool verbose)
{

    // Matrix will get too large, only print if small enough to fit on terminal
    bool printMatrixData = p < 502 ? true : false;

    printf("GPU based mceliece cryptosystem test\n");

    printf("Starting Encryption...\n");
    clock_t start, end;
    double time_used;
    start = clock();

    clock_t mceliece_init_start, mceliece_init_end;
    double mceliece_init_time_used;

    clock_t encrypt_start, encrypt_end;
    double encrypt_time_used;

    clock_t decrypt_start, decrypt_end;
    double decrypt_time_used;


    mceliece_init_start = clock();

    mcc crypt;
    if (cpu)
        crypt = mceliece_init_cpu(n0, p, w, t, seed);
    else {
        crypt = mceliece_init_gpu(n0, p, w, t, seed);
    }


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

    if (verbose && printMatrixData) {
        printf("message:\n");
        for (int i = 0; i < msg->cols; i++)
            printf("%hu", msg->data[i]);
        printf("\n");

        printf("public key:\n");
        for (int i = 0; i < crypt->public_key->cols; i++)
            printf("%hu", crypt->public_key->data[i]);
        printf("\n");
    }


    // error to add to the code
    bin_matrix error = get_error_vector_cpu(crypt->code->n, crypt->code->t);

    if (verbose && printMatrixData) {
        /* display error vector */
        if (cpu) {
            printf("error vector:\n");
            for (int i = 0; i < error->cols; i++) 
                printf("%hu", error->data[i]);
        }
        printf("\n");
    }


    encrypt_start = clock();

    bin_matrix v;
    
    if (cpu)
        v = encrypt_cpu(msg, crypt);
    else
        v = encrypt_gpu(msg, crypt);




    encrypt_end = clock();
    encrypt_time_used = ((double) (encrypt_end - encrypt_start))
        / CLOCKS_PER_SEC;


    if (verbose && printMatrixData) {
        /* display encrypted data */
        printf("encrypted data (message * public key + error):\n");
        for (int i = 0; i < v->cols; i++)
            printf("%hu", v->data[i]);
        printf("\n");
    }


    decrypt_start = clock();
    bin_matrix s;
    if (cpu)
        s = decrypt_cpu(v, crypt);
    else
        s = decrypt_gpu(v, crypt);

    decrypt_end = clock();
    decrypt_time_used = ((double) (decrypt_end - decrypt_start))
        / CLOCKS_PER_SEC;


    if (verbose && printMatrixData) {
        printf("decrypted data:\n");
        for (int i = 0; i < s->cols; i++)
            printf("%hu", s->data[i]);
        printf("\n");
    }

    end = clock();
    time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    if(mat_is_equal_cpu(msg, s)) {
            printf("Decryption %ssuccessful%s...\n", GREEN, NC);
    } else {
            printf("Failure....\n");
    }
    delete_mceliece_cpu(crypt);

    printf("Time taken by cryptosystem (s): %f\n", time_used);
    printf("Time taken by individual phases:\n");
    printf("\tKey Generation: %f - %.2f%%\n", mceliece_init_time_used, 100* (mceliece_init_time_used/time_used));
    //printf("\terror vector generation time: %f - %.2f\%\n",
    //        error_vector_time_used, 100* (error_vector_time_used/time_used));
    printf("\tEncryption:     %f - %.2f%%\n", encrypt_time_used, 100*(encrypt_time_used/time_used));
    printf("\tDecryption:     %f - %.2f%%\n", decrypt_time_used, 100*(decrypt_time_used/time_used));

    free(v->data);
    free(v);
    free(s->data);
    free(s);
    free(msg->data);
    free(msg);
    free(error->data);
    free(error);

    return;
}


#endif /* HAMC_E2E_H */
