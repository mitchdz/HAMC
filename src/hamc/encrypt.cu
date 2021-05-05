#ifndef ENCRYPT_KERNEL_H
#define ENCRYPT_KERNEL_H

#include <stdio.h>
#include <time.h>


#include "TransposeMatrix.cu"
#include "MultiplyMatrix.cu"
#include "hamc_common.h"
#include "hamc_cpu_code.c"
#include "MatrixAdd.cu"


bin_matrix encrypt_gpu(bin_matrix msg, mcc crypt)
{
    if(msg->cols != crypt->public_key->rows) {
        printf("Length of message is incorrect.\n");
        return NULL;
    }
    bin_matrix error = get_error_vector_cpu(crypt->code->n, crypt->code->t);
    bin_matrix word = run_matrix_add_kernel(run_mult_kernel(msg, crypt->public_key), error);
    return word;
}

void generate_message(const char* outputFileName, int k)
{
    if (outputFileName == NULL) {
        printf("ERROR: outputFileName is NULL\n");
        return;
    }

    printf("Generating a message...\n");
    bin_matrix msg = mat_init_cpu(1, k);
    //Initializing the message to be a random message
    for(int i = 0; i < k; i++)
    {
        HAMC_DATA_TYPE_t z = rand() % 2;
        set_matrix_element_cpu(msg, 0, i, z);
    }


    print_bin_matrix(msg);

    write_file_bin_matrix(msg, outputFileName);

    printf("Message generated and stored in %s\n", outputFileName);

    delete_bin_matrix(msg);
}


void run_encryption_from_key(const char* messagePath, const char* publicKeyPath,
    const char* outputPath, int n, int t, bool cpu, bool verbose)
{
    printf("Running encryption...\n");
    bin_matrix msg = read_file_store_bin_matrix(messagePath);
    printf("message is %d rows and %d cols\n", msg->rows, msg->cols);
    //print_bin_matrix(msg);

    printf("Done!");

    bin_matrix G = read_file_store_bin_matrix(publicKeyPath);

    bin_matrix error = get_error_vector_cpu(n, t);

    clock_t encryption_start = clock();

    print_bin_matrix(msg);


    bin_matrix cipher;
    if (cpu)
        cipher = add_matrix_cpu(matrix_mult_cpu(msg, G), error);
    else
        cipher = add_matrix_cpu(run_mult_kernel(msg, G), error);

    clock_t encryption_end = clock();
    double encryption_time = ((double) (encryption_end - encryption_start)) 
        / CLOCKS_PER_SEC;

    write_file_bin_matrix(cipher, outputPath);

    if (verbose) {
        if (cpu)
            printf("Time for encryption (CPU): %lf\n", encryption_time);
        else
            printf("Time for encryption (GPU): %lf\n", encryption_time);
    }

    if (msg) delete_bin_matrix(msg);
    if (G) delete_bin_matrix(G);
    if (error) delete_bin_matrix(error);
    if (cipher) delete_bin_matrix(cipher);
}


void run_encryption(const char* inputFileName, const char* outputFileName,
        int n, int p, int t, int w, int seed, bool cpu)
{
    mcc crypt;
    bin_matrix msg, cipher;
    long f_size;
    int c;
    size_t icc = 0;

    /* open input and output files */
    FILE *in_file  = fopen(inputFileName, "r");

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
    }

    if (cpu)
        cipher = encrypt_cpu(msg, crypt);
    else
        cipher = encrypt_gpu(msg, crypt);

    write_file_bin_matrix(cipher, outputFileName);

    fclose(in_file);
}

#endif /* ENCRYPT_KERNEL_H */
