#include <cuda_runtime.h>
#include <stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>

#include "hamc_cpu_code.c"

#include "hamc_common.h"
#include "decrypt.cu"
#include "encrypt.cu"
#include "keygen.cu"
#include "hamc_e2e.cu"


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

    /* input parameters */
    int n = 2, p = 500, w = 30, t = 10, seed = 10;
    char *outputFileName = NULL, *inputFileName = NULL, *action = NULL;

    /* determines whether to run CPU based implementation, default no */
    bool cpu = false;

    bool verbose = false;

    int c;
    opterr = 0;
    while ((c = getopt (argc, argv, "a:n:p:w:t:i:o:hs:cs:v")) != -1)
        switch(c)
        {
            case 'v':
                verbose = true;
                break;
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

    bool test = false;
    if (!strcmp(action, (const char*)"test")) test = true;

    /* print input parameters */
    printf("\n");
    printf("Input Parameters:\n");
    if (!test) printf("\tInput file: %s%s%s\n", YELLOW, inputFileName, NC);
    if (!test) printf("\tOutput file: %s%s%s\n", YELLOW, outputFileName, NC);
    printf("\tGPU based execution: ");
    if (!cpu) printf("%son%s\n", GREEN, NC);
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
        if (cpu) 
            run_keygen_cpu(n, p, t, w, seed);
        else 
            run_keygen_gpu(n, p, t, w, seed);
    }
    else if (!strcmp(action, (const char*)"encrypt")) {
        if (cpu) 
            run_encryption_cpu(inputFileName, outputFileName, n, p, t, w, seed);
        else 
            run_encryption_gpu(inputFileName, outputFileName, n, p, t, w, seed);
    }
    else if (!strcmp(action, (const char*)"decrypt")) {
        if (cpu) 
            run_decryption_cpu(inputFileName, outputFileName, n, p, t, w, seed);
        //else 
            //decrypt_gpu(outputFileName, n, p, t, w, seed);
    }
    else if (test) {
        test_hamc_e2e(n, p, t, w, seed, cpu, true);
    }
    else {
        printf("action %s not recognized\n", action);
    }
}

void printHelp(){
    printf("\n\nHAMC - Hardware Accelerated Mceliece Cryptosystem\n\n");

    printf("Run the program as such:\n");
    printf("  ./hamc [arguments]\n\n");

    printf("Available Arguments:\n");
    printf("[X] denotes that an argument is required\n");
    printf("\t-a [X] : actions: keygen encrypt decrypt test\n\n");
    printf("\t-c : Run CPU based execution\n\n");
    printf("\t-h : Print this help menu\n\n");
    printf("\t-i [X] : input filename\n\n");
    printf("\t-n [X] : Weight of generator matrix rows \n\n");
    printf("\t-o [X] : output filename\n\n");
    printf("\t-p [X] : Size of matrix during key generation\n\n");
    printf("\t-s : Seed for random number generation\n\n");
    printf("\t-t [X] : Weight of Error Matrix rows\n\n");
    printf("\t-v : Verbose\n\n");
    printf("\t-w [X] : Weight of QC_MDPC code\n\n");

    printf("Example program execution:\n");
    printf("  ./hamc -a test -n 2 -p 1024 -t 10 -w 30 -s 10\n");
    printf("  ./hamc -a test -n 2 -p 500 -t 10 -w 30 -s 10\n");
    printf("  ./hamc -a test -n 2 -p 500 -t 10 -w 30 -s 10 -c\n");


}
