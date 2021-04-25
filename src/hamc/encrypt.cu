#ifndef ENCRYPT_KERNEL_H
#define ENCRYPT_KERNEL_H

#include <stdio.h>
#include <time.h>


#include "TransposeMatrix.cu"
#include "MultiplyMatrix.cu"
#include "hamc_common.h"
#include "hamc_cpu_code.c"



bin_matrix encrypt_gpu(bin_matrix msg, mcc crypt)
{
    if(msg->cols != crypt->public_key->rows) {
        printf("Length of message is incorrect.\n");
        exit(0);
    }
    bin_matrix error = get_error_vector_cpu(crypt->code->n, crypt->code->t);
    bin_matrix word = add_matrix_cpu(run_mult_kernel(msg, crypt->public_key, 16), error);
    return word;
}


// Below function is not in use. Do not assume it works.
void run_encryption_gpu(const char* inputFileName, const char* outputFileName,
        int n, int p, int t, int w, int seed)
{
    mcc crypt;
    bin_matrix msg, m;
    long f_size;
    int c;
    size_t icc = 0;

    /* open input and output files */
    FILE *in_file  = fopen(inputFileName, "r");
    FILE *out_file  = fopen(outputFileName, "w");

    // test for input file not existing
    if (in_file == NULL) {
        printf("Error! Could not open file\n");
        exit(-1);
    }

    /* determine filesize */
    fseek(in_file, 0, SEEK_END);
    f_size = ftell(in_file);
    fseek(in_file, 0, SEEK_SET);

    HAMC_DATA_TYPE_t *input_message = (HAMC_DATA_TYPE_t *)malloc(f_size);

    /* read and write file data into local array */
    icc = 0;
    while ((c = fgetc(in_file)) != EOF) {
        input_message[icc++] = (HAMC_DATA_TYPE_t)c;
    }
    input_message[icc] = '\0';


    printf("\n");
    printf("Input message (char):\n");
    for (int i = 0; i < (int)icc; i++)
        printf("%c",  input_message[i]);
    printf("\n");

    printf("\n");

    /* check that input file is within size */
    int k = (n - 1) * p;
    if ((int)icc > k) {
        printf("ERROR: intput message is too long for k\n");
        printf("input message is length %d while k is %d\n", (int)icc, k);
    }


    /* set up basic encryption primitives */
    crypt = mceliece_init_cpu(n, p, w, t, seed);
    msg = mat_init_cpu(1, k);
    for(int i = 0; i < k; i++) {\
        msg->data[i] = input_message[i];
        //set_matrix_element_cpu(msg, 0, i, (HAMC_DATA_TYPE_t)strtoul(input_message, NULL, 0));
    }

    printf("\n");
    /* run CPU based encryption code */
    m = encrypt_cpu(msg, crypt);
    printf("Encrypted data (HAMC_DATA_TYPE_t):\n");
    for (int i = 0; i < m->cols; i++)
        printf("%hu", m->data[i]);
    printf("\n");

    /* decrypt the ciphertext */
    bin_matrix d = decrypt_cpu(m, crypt);

    printf("Decrypted text:\n");
    for(int i = 0; i < d->cols; i++)
        printf("%c", (char)d->data[i]);


    /* write cipher.text to file */
    for( int i = 0; i < m->cols; i++) {
        fprintf(out_file, "%hu", get_matrix_element_cpu(m, 0, i));
    }

    fclose(in_file);
    fclose(out_file);
}

#endif /* ENCRYPT_KERNEL_H */
