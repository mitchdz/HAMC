#ifndef ENCRYPT_KERNEL_H
#define ENCRYPT_KERNEL_H

#include <stdio.h>
#include <time.h>


#include "TransposeMatrix.cu"
#include "MultiplyMatrix.cu"
#include "hamc_cpu_code.c"


void run_encryption_gpu(const char* inputFileName, const char* outputFileName,
        int n, int p, int t, int w, int seed)
{
    //TODO:
}

#endif /* ENCRYPT_KERNEL_H */
