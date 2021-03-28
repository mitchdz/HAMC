#include "hamc_common.h"

//Reset all positions in the row to 0
void reset_row_cpu(unsigned short* row, int min, int max)
{
    int i;
    for(i = min; i < max + 1; i++) {
        row[i] = 0;
    }
}

//Returns a random integer in the range [min, max]
int random_val(int min, int max, unsigned seed)
{
    int r;
    const unsigned int range = 1 + max - min;
    const unsigned int buckets = RAND_MAX / range;
    const unsigned int limit = buckets * range;

    do {
        r = rand();
    } while (r >= limit);

    return min + (r / buckets);
}


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
