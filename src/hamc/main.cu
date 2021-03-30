#include <cuda_runtime.h>
#include <stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>

#include "hamc_cpu_code.c"

#include "decrypt.cu"
#include "encrypt.cu"
#include "hamc_common.cu"
#include "HAMC_decrypt.cu"
#include "HAMC_encrypt.cu"
#include "HAMC_key_gen.cu"
#include "InverseMatrix.cu"
#include "keygen.cu"
#include "MatrixAdd.cu"
#include "matrix.cu"
#include "mceliece.cu"
#include "MultiplyMatrix.cu"
#include "qc_mdpc.cu"
#include "TransposeMatrix.cu"


#define RED "\033[0;31m"
#define YELLOW "\033[0;33m"
#define GREEN "\033[0;32m"
#define NC "\033[0;0m"


#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

void printHelp();

int main(int argc, char *argv[]) {
    /* variables for timing operations */
    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;

    /* input parameters */
    int action = -1, n = 2, p = 4800, w = 90, t = -1, seed = -1;
    char *outputFileName = NULL, *inputFileName = NULL;
    /* determines whether to run CPU based implementation */
    bool cpu = false;

    printf("HAMC Version 0.1\n");

    int c;
    opterr = 0;
    while ((c = getopt (argc, argv, "a:n:p:w:t:i:o:hs:cs:")) != -1)
        switch(c)
        {
            case 'c':
                cpu = true;
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'p':
                p = atoi(optarg);
                break;
            case 'w':
                w = atoi(optarg);
                break;
            case 't':
                t = atoi(optarg);
                break;
            case 'i':
                inputFileName = strdup(optarg);
                //strcpy(inputFileName, (const char*)optarg);
                break;
            case 'o':
                outputFileName = strdup(optarg);
                //strcpy(outputFileName, (const char*)optarg);
                break;
            case 'a':
                action = atoi(optarg);
                break;
            case 'h':
                printHelp();
                return(1);
            default:
                abort();

        }

    int k = (n - 1) * p;

    /* print input parameters */
    printf("Input Parameters:\n");
    printf("Input file: %s%s%s\n", YELLOW, inputFileName, NC);
    printf("Output file: %s%s%s\n", YELLOW, outputFileName, NC);
    printf("cpu based execution: ");
    if (cpu) printf("%son%s\n", GREEN, NC);
    else printf("%soff%s\n", RED, NC);
    printf("n: %s%d%s\n", YELLOW, n, NC);
    printf("p: %s%d%s\n", YELLOW, p, NC);
    printf("w: %s%d%s\n", YELLOW, w, NC);
    printf("t: %s%d%s\n", YELLOW, t, NC);
    printf("k: %s%d%s\n", YELLOW, k, NC);
    printf("seed: %s%d%s\n", YELLOW, seed, NC);
    printf("action: %s", YELLOW);
    if (action == 1)
        printf("keygen\n");
    else if (action == 2)
        printf("encrypt\n");
    else if (action == 3)
        printf("decrypt\n");
    else
        printf("unkown\n");
    printf("%s", NC);


    /* 1)keygen 2) encrypt 3)decrypt */
    switch(action)
    {
        case 1: //keygen
            if (cpu) run_keygen_cpu(outputFileName, n, p, t, w, seed);
            else run_keygen_gpu(outputFileName, n, p, t, w, seed);
            break;
        case 2: //encrypt
            if (cpu)
                run_encryption_cpu(inputFileName, outputFileName, n, p, t, w,
                        seed);
            else run_encryption_gpu(inputFileName, outputFileName, n, p, t, w,
                    seed);
            break;
        case 3: //decrypt
            if (cpu) run_decryption_cpu(inputFileName, outputFileName, n, p, t, w, seed);
            else run_decryption_gpu(outputFileName, n, p, t, w, seed);
            break;
        default:
            printf("Wrong action given to system.\n");
            printHelp();
            return(1);
    }
}

void printHelp(){
    printf("HAMC - Hardware Accelerated Mceliece Cryptosystem\n");
    printf("Usage:\n");
    printf("./hamc <arguments>\n");
    printf("Available Arguments:\n");
    printf("\ta (REQUIRED)\n");
    printf("\t\t - action: 1)keygen 2)encrypt 3)decrypt\n");

    printf("\tn\n");
    printf("\t\t - \n");

    printf("\tp\n");
    printf("\t\t - \n");

    printf("\tw\n");
    printf("\t\t - \n");

    printf("\tt\n");
    printf("\t\t - \n");

    printf("\ti\n");
    printf("\t\t - input filename\n");

    printf("\to\n");
    printf("\t\t - output filename\n");
}
