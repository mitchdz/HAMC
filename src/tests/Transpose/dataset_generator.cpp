#ifndef HAMC_TRANSPOSE_DATASET_GENERATOR_H
#define HAMC_TRANSPOSE_DATASET_GENERATOR_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "wb.h"
#include "../../hamc/hamc_cpu_code.c"
#include "../../hamc/hamc_common.h"

static char *base_dir;

static void compute(HAMC_DATA_TYPE_t *output, HAMC_DATA_TYPE_t *input0, int numARows, int numACols)
{
    bin_matrix A = mat_init_cpu(numARows, numACols);

    A->data = input0;

    bin_matrix B = transpose_cpu(A);
    for(int i = 0; i < numARows * numACols; i++){
        output[i] = B->data[i];
    }

    free(A);
}

static HAMC_DATA_TYPE_t *generate_data(int height, int width)
{
    HAMC_DATA_TYPE_t *data = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t) * width * height);
    int i;
    for (i = 0; i < width * height; i++) {
        data[i] = (HAMC_DATA_TYPE_t)(rand() % 2); // 0 or 1
    }
    return data;
}

static void write_data(char *file_name, HAMC_DATA_TYPE_t *data, int height, int width)
{
    int ii, jj;
    FILE *handle = fopen(file_name, "w");
    fprintf(handle, "%d %d\n", height, width);
    for(ii = 0; ii < height; ii++){
        for(jj = 0; jj < width; jj++){
            fprintf(handle, "%.2hi", *data++);
            if(jj != width - 1){
                fprintf(handle, " ");
            }
        }
        if(ii != height - 1){
            fprintf(handle, "\n");
        }
    }
    fflush(handle);
    fclose(handle);
}

static void create_dataset(int datasetNum, int numARows, int numACols)
{
    const char *dir_name =
        wbDirectory_create(wbPath_join(base_dir, datasetNum));

    char *input0_file_name = wbPath_join(dir_name, "input0.raw");
    char *output_file_name = wbPath_join(dir_name, "output.raw");

    HAMC_DATA_TYPE_t *input0_data = generate_data(numARows, numACols);
    HAMC_DATA_TYPE_t *output_data = (HAMC_DATA_TYPE_t *)calloc(sizeof(HAMC_DATA_TYPE_t), numARows * numACols);

    compute(output_data, input0_data, numARows, numACols);

    write_data(input0_file_name, input0_data, numARows, numACols);
    write_data(output_file_name, output_data, numACols, numARows);

    free(input0_data);
    free(output_data);
}

int main()
{
    base_dir = wbPath_join(wbDirectory_current(), "Transpose", "Dataset");

    create_dataset(0, 32, 32);
    create_dataset(0, 16, 16);
    create_dataset(1, 64, 64);
    create_dataset(2, 64, 128);
    create_dataset(3, 112, 48);
    create_dataset(4, 84, 84);
    create_dataset(5, 80, 99);
    create_dataset(6, 128, 128);
    create_dataset(7, 100, 100);
    create_dataset(8, 134, 130);
    create_dataset(9, 417, 210);
    create_dataset(10, 500, 500);
    create_dataset(11, 2000, 2000);
  return 0;
}

#endif /* HAMC_TRANSPOSE_DATASET_GENERATOR_H */
