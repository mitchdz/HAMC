#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "wb.h"
#include "../../hamc/hamc_cpu_code.c"

static char *base_dir;

static void compute(HAMC_DATA_TYPE_t *output, HAMC_DATA_TYPE_t *input0, HAMC_DATA_TYPE_t *input1, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
{
    bin_matrix A = mat_init_cpu(numARows, numACols);
    bin_matrix B = mat_init_cpu(numBRows, numBCols);
    bin_matrix C = mat_init_cpu(numCRows, numCCols);
    
    A->data = input0;
    B->data = input1;
    
    C = matrix_mult_cpu(A, B);
    for(int i = 0; i < numCRows * numCCols; i++){
        output[i] = C->data[i];
    }
    
    free(A);
    free(B);
    free(C);
}

static HAMC_DATA_TYPE_t *generate_data(int height, int width)
{
    HAMC_DATA_TYPE_t *data = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t) * width * height);
    int i;
    for (i = 0; i < width * height; i++) {
        data[i] = (HAMC_DATA_TYPE_t)(rand() % 2);
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

static void create_dataset(int datasetNum, int numARows, int numACols, int numBCols)
{
    int numBRows = numACols;
    int numCRows = numARows;
    int numCCols = numACols;

    const char *dir_name = wbDirectory_create(wbPath_join(base_dir, datasetNum));

    char *input0_file_name = wbPath_join(dir_name, "input0.raw");
    char *input1_file_name = wbPath_join(dir_name, "input1.raw");
    char *output_file_name = wbPath_join(dir_name, "output.raw");

    HAMC_DATA_TYPE_t *input0_data = generate_data(numARows, numACols);
    HAMC_DATA_TYPE_t *input1_data = generate_data(numBRows, numBCols);
    HAMC_DATA_TYPE_t *output_data = (HAMC_DATA_TYPE_t *)calloc(sizeof(HAMC_DATA_TYPE_t), numCRows * numCCols);

    compute(output_data, input0_data, input1_data, numARows, numACols, numBRows, numBCols, numCRows, numCCols);
    
    write_data(input0_file_name, input0_data, numARows, numACols);
    write_data(input1_file_name, input1_data, numBRows, numBCols);
    write_data(output_file_name, output_data, numCRows, numCCols);

    free(input0_data);
    free(input1_data);
    free(output_data);
}

int main()
{
    base_dir = wbPath_join(wbDirectory_current(), "MatrixMultiply", "Dataset");

    create_dataset(0, 16, 16, 16);
    create_dataset(1, 64, 64, 64);
    create_dataset(2, 64, 128, 64);
    create_dataset(3, 112, 48, 16);
    create_dataset(4, 84, 84, 84);
    create_dataset(5, 80, 99, 128);
    create_dataset(6, 128, 128, 128);
    create_dataset(7, 100, 100, 100);
    create_dataset(8, 134, 130, 150);
    create_dataset(9, 500, 500, 500);
  return 0;
}
