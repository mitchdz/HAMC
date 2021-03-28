#include "hamc_common.h"
#include "qc_mdpc.cu"
#include "mceliece.h"
#include "hamc_common.h"
void* safe_malloc(size_t n)
{
    void* p = malloc(n);
    if (!p)
    {
        fprintf(stderr, "Out of memory(%lu bytes)\n",(size_t)n);
        exit(EXIT_FAILURE);
    }
    return p;
}


void run_encryption_gpu(const char* inputFileName, const char* outputFileName, 
        int n, int p, int t, int w, int seed)
{
    //TODO: implement
}

void run_encryption_cpu(const char* inputFileName, const char* outputFileName,
        int n, int p, int t, int w, int seed)
{
    cudaEvent_t astartEvent, astopEvent;

    int numARows, numAColumns;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);

    // retrieve message from file
    /* wbImport only reads and writes float, so we need to convert that */
    float *hostAFloats = (float *)wbImport(inputFileName, &numARows, &numAColumns);
    ushort *hostA = (ushort *)malloc(numARows*numAColumns * sizeof(ushort));
    for (int i = 0; i < numARows*numAColumns; i++)
        hostA[i] = (ushort)hostAFloats[i];

    // initialize encryption algo
    mcc crypt;
    crypt = (mcc)safe_malloc(sizeof(struct mceliece));
    crypt->code = qc_mdpc_init_cpu(n, p, w, t);
    crypt->public_key = generator_matrix_cpu(crypt->code);
    //printf("mceliece generated...\n");

    /* get error vector */

    /* add message, public key, and error */

    /* determine error length and weight */
    int error_length = 0;
    int error_weight = 0;

    ushort *error = get_error_vector_cpu(error_length, error_weight)->data;

}


