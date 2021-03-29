#include <cuda_runtime.h>
#include<stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>

#include "Multiply_cpu.cpp"
#include "../../hamc/MultiplyMatrix.cu"

#define TILE_WIDTH 16
#define ushort unsigned short

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess){
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define mat_element(mat, row_idx, col_idx) \
    mat->data[row_idx * (mat->cols) + col_idx]

typedef struct matrix
{
   int rows;             //number of rows.
   int cols;             //number of columns.
   ushort *data;
}*bin_matrix;

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

bin_matrix mat_init(int rows, int cols){
	if(rows <= 0 || cols <= 0)
		return NULL;

	bin_matrix A;
	A = (bin_matrix)safe_malloc(sizeof(struct matrix));
	A->cols = cols;
	A->rows = rows; 
	A->data = (ushort *)safe_malloc(rows*cols*sizeof(ushort)); 
	return A;
}

void printHelp()
{
    printf("run this executable with the following flags\n");
    printf("\n");
    printf("\t-a <input0 file name>\n");
    printf("\t-b <input1 file name>\n");
    printf("\t-e <expected solution file name>\n");
    printf("\t-o <output file name>\n");
    printf("\t-c \n");
}


bin_matrix run_cpu(bin_matrix A, bin_matrix B)
{
    return mult_matrix(A, B);
}

bin_matrix run_kernel(bin_matrix A, bin_matrix B)
{
    if (A->cols != B->rows){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }

    ushort *deviceA;
    ushort *deviceB;
    ushort *deviceC;
    
    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(ushort));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(ushort));
    cudaMalloc((void **) &deviceC, B->cols * A->rows * sizeof(ushort));
    
    cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(ushort), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(ushort), cudaMemcpyHostToDevice);
    
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((B->cols - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);
    
    matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(C->data, deviceC, B->cols * A->rows * sizeof(ushort), cudaMemcpyDeviceToHost);
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}

int main(int argc, char *argv[])
{
    wbArg_t args;
    bin_matrix A;
    bin_matrix B;
    bin_matrix C;
    int numRowsA;
    int numColsA;
    int numRowsB;
    int numColsB;
    int numRowsS;
    int numColsS;
    ushort *hostA;
    ushort *hostB;
    ushort *sol;
    char *input0;
    char *input1;
    char *expected;
    bool cpu_exec = false;
    bool solved = true;
    
    int opt;
    while ((opt = getopt (argc, argv, "a:b:e:o:c")) != -1){
        switch(opt){
            case 'a':
                input0 = strdup(optarg);
                break;
            case 'b':
                input1 = strdup(optarg);
                break;
            case 'e':
                expected = strdup(optarg);
                break;
            case 'o':
                //input0 = strdup(optarg);
                break;
            case 'c':
                cpu_exec = true;
                break;
            case 'h':
            default:
                printHelp();
                return 0;
        }
    }
    
    hostA = (ushort *)wbImport(input0, &numRowsA, &numColsA);
    A = mat_init(numRowsA, numColsA);
    A->data = hostA;
    hostB = (ushort *)wbImport(input1, &numRowsB, &numColsB);
    B = mat_init(numRowsB, numColsB);
    B->data = hostB;
    sol = (ushort *)wbImport(expected, &numRowsS, &numColsS);
    
    if(cpu_exec){
        C = run_cpu(A, B);
    }
    else{
        C = run_kernel(A, B);
    }
    //C = (cpu_exec) ? run_cpu(A, B) : run_kernel(A, B);
    
    if(C->rows != S->rows && C->cols != S->cols){
        solved = false;
    }
    else{
        for(int i = 0; i < numRowsS * numColsS; i++){
            if(C->data[i] != S->data[i]){
                solved = false;
                break;
            }
        }
    }
    
    cout << solved << endl;
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}