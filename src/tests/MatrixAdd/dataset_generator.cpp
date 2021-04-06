#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "wb.h"
#include "../../hamc/hamc_cpu_code.c"

#define ushort unsigned short

static char *base_dir;

static ushort *generate_data(int height, int width) {
  ushort *data = (ushort *)malloc(sizeof(ushort) * width * height);
  int i;
  for (i = 0; i < width * height; i++) {
    //data[i] = ((ushort)(rand() % 20) - 5) / 5.0f;
    data[i] = (ushort)(rand() % 2);
  }
  return data;
}

static void write_data(char *file_name, ushort *data, int height,
                       int width) {
  
  int ii, jj;
  FILE *handle = fopen(file_name,"w");
  
  fprintf(handle, "%d %d\n", height, width);
  
  for (ii = 0; ii < height; ii++) {
    for (jj = 0; jj < width; jj++) {
     fprintf(handle, "%.2hi", *data++);
      if (jj != width - 1) fprintf(handle, " ");
    }
    if (ii != height - 1) fprintf(handle, "\n");
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, int numARows, int numACols,int numBRows, int numBCols) {
    int numCRows = numARows;
    int numCCols = numACols;
    
    const char *dir_name =
        wbDirectory_create(wbPath_join(base_dir, datasetNum));
        
   

    char *input0_file_name = wbPath_join(dir_name, "input0.raw");
    
    char *input1_file_name = wbPath_join(dir_name, "input1.raw");
    
    char *output_file_name = wbPath_join(dir_name, "output.raw");
  
    ushort *input0_data = generate_data(numARows, numACols);
    ushort *input1_data = generate_data(numBRows, numBCols);
    ushort *output_data = (ushort *)calloc(sizeof(ushort), numCRows * numCCols);

    // Create Matricies
    
    bin_matrix A = mat_init_cpu(numARows, numACols);
    A->data = input0_data;
    
    bin_matrix B = mat_init_cpu(numBRows, numBCols);
    B->data = input1_data;
    
  
    bin_matrix output = matrix_add_cpu(A,B);

    
    
    write_data(input0_file_name, input0_data, numARows, numACols); 
    
    write_data(input1_file_name, input1_data, numBRows, numBCols);

    write_data(output_file_name, output->data, output->rows, output->cols);


    free(input0_data);
    free(input1_data);
    free(output_data);
}

int main() {
  base_dir = wbPath_join(wbDirectory_current(),
                         "MatrixAdd", "Dataset");
    
   
  create_dataset(0, 16, 16, 16, 16);
  create_dataset(1, 4, 4, 4, 4);
  create_dataset(2, 500, 500, 500, 500);
  create_dataset(3, 1000, 1000, 1000, 1000);
  create_dataset(4, 500, 1000, 500, 1000);
  return 0;
}
