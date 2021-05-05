#ifndef ADD_KERNEL_H
#define ADD_KERNEL_H

#include "hamc_cpu_code.c"
#include "hamc_common.h"

#define BLOCK_DIM 16
#define ADD_TILE_WIDTH 16

__global__ void MatrixAdd(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C,
        int rows, int cols);

/* 	run_matrix_add_kernel - this function calls the kernel to element-wise
*	add both matrices A and B together
*	param 1 - bin_matrix A - containes the rows, columns, and data for matrix A
* 	param 2 - bin_matrix B - contains the rows, columns, and data for matrix B
*/
bin_matrix run_matrix_add_kernel(bin_matrix A, bin_matrix B)
{
	// Conditional - A and B have to have the same rows and columns
	// 		otherwise, matrix addition is impossible.
    if (A->rows != B->rows || A->cols != B->cols){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }
    
	// Declare Device Variables
    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;
    HAMC_DATA_TYPE_t *deviceC;
    
	// Instantiate and intialize the output matrix C with having the same number
	// rows as A and same number of columns of B.
    bin_matrix C = mat_init_cpu(A->rows,B->cols);

	// Dynamically allocate memory for device variables
    cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceB, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceC, B->cols * A->rows * sizeof(HAMC_DATA_TYPE_t));

	// Copy the matrix data from A and B matrices respectively from Host variables side to Device variables  
    cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);
   
	// Designed the Blocks Dimiension to ADD_TILE_WIDTH (16)
    dim3 DimBlock(ADD_TILE_WIDTH, ADD_TILE_WIDTH, 1);
	
	// Calculations to figure out how many tiles there needs to be in X and Y dimensions.
    int x_blocks = ((B->cols - 1)/ADD_TILE_WIDTH) + 1;
    int y_blocks = ((A->rows - 1)/ADD_TILE_WIDTH) + 1;
	
	// Define grid dimension from previous calculations
    dim3 DimGrid(x_blocks, y_blocks, 1);
  
	// MAKE KERNEL CALL
    MatrixAdd<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, A->rows, A->cols);

    //printf("\n");
    //printf("Total compute time (ms) %f for Matrix Add GPU\n\n",aelapsedTime);
    //printf("\n");
    
	// Copy resulting output from kernel call from device back to host memory (into C matrix data array)
    cudaMemcpy(C->data, deviceC, B->cols * A->rows * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);

	// Free device memory variables
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
   
	// Return C bin_matrix.
    return C;
}


/*	MatrixAdd - Globally reads from input matrix A and matrix B and writes back into matrix C
*	HAMC_DATA_TYPE_t - as specified in hamc_common.h was formerly, the original unsigned short
*			but is now uint_8.
*	rows, cols - since matrix A and B must have the same columns, we can use these variables to
*			prevent segmentation faults and confine thread access to the bounds.
*/
__global__ void MatrixAdd(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C,
        int rows, int cols) {
        
	// Finds defines thread index in both x and y dimensions
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Finds the output address into the output matrix C
    int index = row * cols + col;
 
	// Conditional - Prevents segmentation fault by preventing accesses outside bounds of A
	// 		and B matrices.
    if((row < rows) && (col < cols)) {
	    // Performs binary matrix addition (XOR) without carry-over
  	    C[index] = A[index] ^ B[index];
    }

}
#endif // ADD_KERNEL_H
