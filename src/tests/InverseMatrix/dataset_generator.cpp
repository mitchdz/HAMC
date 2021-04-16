#ifndef HAMC_TRANSPOSE_DATASET_GENERATOR_H
#define HAMC_TRANSPOSE_DATASET_GENERATOR_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "wb.h"
#include "../../hamc/hamc_cpu_code.c"
#include "../../hamc/hamc_common.h"

static char *base_dir;

//Inverse of matrix
static bool inverse_cpu_check(bin_matrix A)
{
    if(A->rows != A->cols) {
      printf("Inverse not possible...\n");
      exit(0);
    }

    if(is_identity_cpu(A)) {
      return A;
    }

    bin_matrix B;
    B = mat_init_cpu(A->rows, A->cols);
    make_indentity_cpu(B);

    int i;

    for(i = 0; i < A->cols; i++) {
        if(mat_element(A, i, i) == 1) {
            for(int j = 0; j <  A->rows; j++) {
                if(i != j && mat_element(A, j, i) == 1) {
                    add_rows_new_cpu(B, i, j, 0, A->cols);
                    add_rows_new_cpu(A, i, j, i, A->cols);
                }
            }
        }
        else{
            int k;
            for(k = i + 1; k < A->rows; k++) {
                if(mat_element(A, k, i) == 1) {
                    add_rows_cpu(B, k, i);
                    add_rows_cpu(A, k, i);
                    i = i - 1;
                    break;
                }
            }
        }
    }
    //printf("Out of for loop...\n");
    if(!is_identity_cpu(A))
    {
      //printf("Could not find inverse...\n");
      return false;
    }

    for (int i = 0; i < A->rows*A->cols; i++) {
        A->data[i] = B->data[i];
    }

    free(B->data);
    free(B);
    return true;
}
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

static void generate_data(HAMC_DATA_TYPE_t *data1, HAMC_DATA_TYPE_t *data2, int height, int width, int seed)
{
    //HAMC_DATA_TYPE_t *data = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t) * width * height);
    int i;
    srand(seed);
    for (i = 0; i < width * height; i++) {
        HAMC_DATA_TYPE_t val = (HAMC_DATA_TYPE_t)(rand() % 2); // 0 or 1
        data1[i] = val;
        data2[i] = val;
    }
}

static void write_data(char *file_name, HAMC_DATA_TYPE_t *data, int height, int width)
{
    int ii, jj;
    FILE *handle = fopen(file_name, "w");
    fprintf(handle, "%d %d\n", height, width);
    for(ii = 0; ii < height; ii++){
        for(jj = 0; jj < width; jj++){
            fprintf(handle, "%d", *data++);
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

void printMatrix(bin_matrix A)
{
    if (A->rows < 40) {
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < A->cols; j++) {
                printf("%d ", A->data[i*A->cols + j]);
            }
            printf("\n");
        }
    }

}

static void create_dataset(int datasetNum, int n, int p, int t, int w, int seed)
{
    printf("Creating dataset for %dx%d...\n",p, p);
    const char *dir_name =
        wbDirectory_create(wbPath_join(base_dir, datasetNum));

    char *input0_file_name = wbPath_join(dir_name, "input.raw");
    char *output_file_name = wbPath_join(dir_name, "output.raw");

    //HAMC_DATA_TYPE_t *output_data = (HAMC_DATA_TYPE_t *)calloc(sizeof(HAMC_DATA_TYPE_t), numARows * numACols);

    //bin_matrix A = mat_init_cpu(numARows, numACols);
    //bin_matrix B = mat_init_cpu(numARows, numACols);


    clock_t start, end;
    double time_used;

    start = clock();

    bool ret;

    //int maxAttempts = 20000;
    //int i;
    //for (i = 6000; i < maxAttempts; i++) {
    //    printf("seed: %d\n", i);
    //    generate_data(B->data, A->data, numARows, numACols, i);

    //    ret = inverse_cpu_check(B);
    //    if (ret == true) break;
    //}

    mdpc code = qc_mdpc_init_cpu(n, p, t, w, seed);
    bin_matrix A = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 1);
    bin_matrix B = mat_init_cpu(p, p);
    for (int i =0; i < p*p; i++) {
        B->data[i] = A->data[i];
    }
    ret = inverse_cpu_check(A);


    end = clock();
    time_used = ((double) (end - start))/ CLOCKS_PER_SEC;


    printf("Found inverse for %dx%d with seed %d in %lfs\n", p, p, seed, time_used);
    printMatrix(B);

    write_data(input0_file_name, A->data, p, p);
    write_data(output_file_name, B->data, p, p);

    free(A->data);
    free(A);
    free(B->data);
    free(B);
    //free(output_data);
}

int main()
{
    base_dir = wbPath_join(wbDirectory_current(), "inverse", "Dataset");



      //create_dataset(1, 2, 200, 10, 30, 10);
      //create_dataset(2, 2, 500, 10, 30, 10);
      //create_dataset(3, 2, 512, 10, 30, 10);
      //create_dataset(4, 2, 1024, 10, 30, 10);
      //create_dataset(5, 2, 2000, 10, 120, 10);
      //create_dataset(6, 2, 4800, 20, 60, 10);
      //create_dataset(7, 2, 6000, 20, 60, 10);
      //create_dataset(8, 2, 12000, 20, 60, 10);
      create_dataset(9, 2, 24000, 264, 274, 10);
      create_dataset(10, 2, 32771, 264, 274, 10);

  return 0;
}

#endif /* HAMC_TRANSPOSE_DATASET_GENERATOR_H */
