#include <cuda_runtime.h>
#include <stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>

#include "hamc_common.h"
#include "qc_mdpc.h"
#include "keygen.h"
#include "encrypt.cu"
#include "decrypt.cu"

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
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *deviceInput;
    unsigned int *deviceBins;
    int seed;

    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;

    int action, n, p, w, t;
    char *outputFileName, *inputFileName;

    /* determines whether to run CPU based implementation */
    bool cpu = false;

    int c;
    opterr = 0;
    while ((c = getopt (argc, argv, "a:npwtiohs:c")) != -1)
        switch(c)
        {
            case 'c':
                cpu = true;
                break;
            case 'n':
                n = *optarg;
                break;
            case 's':
                seed = *optarg;
                break;
            case 'p':
                p = *optarg;
                break;
            case 'w':
                w = *optarg;
                break;
            case 't':
                t = *optarg;
                break;
            case 'i':
                strcpy(inputFileName, (const char*)optarg);
                break;
            case 'o':
                strcpy(outputFileName, (const char*)optarg);
                break;
            case 'a':
                action = *optarg;
                break; case 'h':
                printHelp();
                return(1);
            default:
                abort();

        }

    int k = (n - 1) * p;

    /* print input parameters */
    printf("Input Parameters:\n");
    printf("Input file: %s\n", inputFileName);
    printf("Output file: %s\n", outputFileName);
    printf("cpu based execution: ");
    if (cpu) printf("on\n");
    else printf("off\n");
    printf("n: %d\n", n);
    printf("p: %d\n", p);
    printf("w: %d\n", w);
    printf("t: %d\n", t);
    printf("seed: %d\n", seed);
    printf("action: ");
    if (action == 1)
        printf("keygen\n");
    else if (action == 2)
        printf("encrypt\n");
    else if (action == 3)
        printf("decrypt\n");
    else
        printf("unkown\n");

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
            if (cpu) run_decryption_cpu(outputFileName, n, p, t, w, seed);
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
