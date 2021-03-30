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
void printWelcome()
{
    printf("HAMC Version %s0.1%s\n", YELLOW, NC);
    printf("Developed by Mitchell Dzurick, Mitchell Russel, James Kuban");
}



int main(int argc, char *argv[]) {
    printWelcome();

    /* variables for timing operations */
    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;

    /* input parameters */
    int n = 2, p = 500, w = 30, t = 10, seed = 10;
    char *outputFileName = NULL, *inputFileName = NULL, *action = NULL;
    /* determines whether to run CPU based implementation */
    bool cpu = false;


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
                action = strdup(optarg);
                break;
            case 'h':
                printHelp();
                return(1);
            default:
                abort();

        }

    int k = (n - 1) * p;

    /* print input parameters */
    printf("\n");
    printf("Input Parameters:\n");
    printf("\tInput file: %s%s%s\n", YELLOW, inputFileName, NC);
    printf("\tOutput file: %s%s%s\n", YELLOW, outputFileName, NC);
    printf("\tcpu based execution: ");
    if (cpu) printf("%son%s\n", GREEN, NC);
    else printf("%soff%s\n", RED, NC);
    printf("\tn: %s%d%s\n", YELLOW, n, NC);
    printf("\tp: %s%d%s\n", YELLOW, p, NC);
    printf("\tw: %s%d%s\n", YELLOW, w, NC);
    printf("\tt: %s%d%s\n", YELLOW, t, NC);
    printf("\tk: %s%d%s\n", YELLOW, k, NC);
    printf("\tseed: %s%d%s\n", YELLOW, seed, NC);
    printf("\taction: %s%s%s\n", YELLOW, action, NC);


    //TODO: make sure action is null-terminated before passing into strcmp
    if (!strcmp(action, (const char*)"keygen")) {
        if (cpu) run_keygen_cpu(outputFileName, n, p, t, w, seed);
        else run_keygen_gpu(outputFileName, n, p, t, w, seed);
    }
    else if (!strcmp(action, (const char*)"encrypt")) {
        if (cpu) run_encryption_cpu(inputFileName, outputFileName, n, p, t, w, seed);
        else run_encryption_gpu(inputFileName, outputFileName, n, p, t, w, seed);
    }
    else if (!strcmp(action, (const char*)"decrypt")) {
        if (cpu) run_decryption_cpu(inputFileName, outputFileName, n, p, t, w, seed);
        else run_decryption_gpu(outputFileName, n, p, t, w, seed);
    }
    else if (!strcmp(action, (const char*)"test")) {
        if (cpu) test_cpu_e2e(n, p, t, w, seed);
        //else (outputFileName, n, p, t, w, seed);
    }
    else {
        printf("action %s not recognized\n", action);
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
