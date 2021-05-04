
#ifndef HAMC_DECRYPT_H
#define HAMC_DECRYPT_H


#include "hamc_common.h"
#include "hamc_cpu_code.c"

#include "MultiplyMatrix.cu"
#include "encrypt.cu"
#include "TransposeMatrix.cu"


// read either public or private key
bin_matrix read_code_from_file()
{
    printf("TODO: read code from file\n");
    return NULL;
}


//Decoding the codeword
bin_matrix decode_gpu(bin_matrix word, mdpc code)
{
    bin_matrix H = parity_check_matrix_cpu(code);

    //bin_matrix syn  = matrix_mult_cpu(H, transpose_cpu(word));
    bin_matrix syn = run_mult_kernel(H, run_transpose_kernel(word));
    
    int limit = 10;
    int delta = 5;
    int i,j,k,x;

    // Syndrome decoding
    for(i = 0; i < limit; i++) {
        //printf("Iteration: %d\n", i);
        int unsatisfied[word->cols];
        for(x = 0; x < word->cols; x++) {
          unsatisfied[x] = 0;
        }

        for(j = 0; j < word->cols; j++) {
            for(k = 0; k < H->rows; k++) {
               if(get_matrix_element_cpu(H, k, j) == 1) {
                   if(get_matrix_element_cpu(syn, k, 0) == 1) {
                       unsatisfied[j] = unsatisfied[j] + 1;
                   }
               }
            }
        }

        // printf("No. of unsatisfied equations for each bit: \n");
        // for(int idx = 0; idx < word->cols; idx++) {
        //     printf("b%d: %d \n", idx, unsatisfied[idx]);
        // }
        int b = get_max_cpu(unsatisfied, word->cols) - delta;
        for(j = 0; j < word->cols; j++) {
            if(unsatisfied[j] >= b) {
                set_matrix_element_cpu(word, 0, j, (get_matrix_element_cpu(word, 0, j) ^ 1));
                syn = add_matrix_cpu(syn, mat_splice_cpu(H, 0, H->rows - 1, j, j));


            }
        }
        // printf("Syndrome: ");
        // print_matrix(syn);
        // printf("\n");
        //printf("Iteration: %d\n", i);
        if(is_zero_matrix_cpu(syn)) {
            return word;
        }
    }


    printf("Decoding failure...\n");
    exit(0);
}

bin_matrix decrypt_gpu(bin_matrix word, mcc crypt)
{
    if(word->cols != crypt->code->n) {
        printf("Length of message is incorrect while decrypting.\n");
        return NULL;
    }
    //printf("Decryption started...\n");
    bin_matrix msg = decode_gpu(word, crypt->code);
    msg = mat_splice_cpu(msg, 0, msg->rows - 1, 0, crypt->code->k - 1);
    return msg;
}


bin_matrix decrypt_from_file_gpu(const char* inputFileName, bin_matrix word, mcc crypt)
{
    // TODO: finish implementation
    bin_matrix B = mat_init_cpu(word->rows, word->cols);
    return B;
}


bin_matrix run_decrypt(bin_matrix word, mcc crypt, bool cpu, bool verbose)
{
    if (verbose) printf("TODO: finish decryption implementation\n");
    bin_matrix msg;

    if (cpu)
        msg = decrypt_cpu(word, crypt);
    else 
        msg = decrypt_gpu(word, crypt);

    if (verbose) printf("Done\n");

    return msg;
}



#endif /* HAMC_DECRYPT_H */
