#ifndef HAMC_KEYGEN_H
#define HAMC_KEYGEN_H

#include <stdio.h>

#include "hamc_cpu_code.c"
#include "hamc_common.h"
#include "mceliece.cu"


void run_keygen(int n, int p, int t, int w, int seed, bool cpu, bool verbose)
{
    mcc crypt = mceliece_init_cpu(n, p, w, t, seed);
    
    // H is not sped up too much from GPU. For now, just run CPU version.
    bin_matrix H = parity_check_matrix_cpu(crypt->code);
    
    bin_matrix G;
    if (cpu)
        G = generator_matrix_cpu(crypt->code);
    else
        G = generator_matrix_gpu(crypt->code);

    FILE *fp1, *fp2;
    fp1 = fopen("Private_Key.txt", "w");
    fprintf(fp1, "%d %d\n", p, p);
    for(int i = 0; i < H->rows; i++) {
        for(int j = 0; j < H->cols; j++) {
            fprintf(fp1, "%hu ", get_matrix_element_cpu(H, i, j));
        }
        fprintf(fp1, "\n");
    }
    fclose(fp1);

    fp2 = fopen("Public_Key.txt", "w");
    fprintf(fp2, "%d %d\n", p, p);
    for(int i = 0; i < G->rows; i++) {
        for(int j = 0; j < G->cols; j++) {
            fprintf(fp2, "%hu ", get_matrix_element_cpu(G, i, j));
        }
        fprintf(fp2, "\n");
    }
    fclose(fp2);
    printf("Keys Generated...\n");
    printf("Private_Key.txt\n");
    printf("Public_Key.txt\n");
}

#endif /* HAMC_KEYGEN_H */
