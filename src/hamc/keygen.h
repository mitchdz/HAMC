#ifndef HAMC_KEYGEN_H
#define HAMC_KEYGEN_H

void run_keygen_gpu(const char* outputFileName, int n, int p, int t, int w,
    int seed);

void run_keygen_cpu(const char* outputFileName, int n, int p, int t, int w,
    int seed);

#endif /* HAMC_KEYGEN_H */
