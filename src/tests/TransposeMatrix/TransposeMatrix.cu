#include "../test.h"

__global__ void transpose(ushort *A, ushort *B, int rowA, int colA) {
	extern __shared__ ushort shared_B[];
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(row < rowA && col < colA){
		shared_B[col * rowA + row] = A[row * colA + col];
    }
	__syncthreads();
	
	if(row < rowA && col < colA){
		B[row * colA + col] = shared_B[row * colA + col];
    }
}

bin_matrix transpose(bin_matrix A) {
	ushort *deviceA;
	ushort *deviceB;
	
	bin_matrix B;
	B = mat_init(A->cols, A->rows);
	
	cudaMalloc((void **) &deviceA, A->cols * A->rows * sizeof(ushort));
	cudaMalloc((void **) &deviceB, A->rows * A->cols * sizeof(ushort));
	
	cudaMemcpy(deviceA, A->data, A->cols * A->rows * sizeof(ushort), cudaMemcpyHostToDevice);
	
	dim3 DimBlock(32, 32, 1);
	int x_blocks = ((A->cols - 1)/32) + 1;
    int y_blocks = ((A->rows - 1)/32) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);
	
	transpose<<<DimGrid, DimBlock, A->cols * A->rows * sizeof(ushort)>>>(deviceA, deviceB, A->rows, A->cols);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(A->data, deviceB, A->cols * A->rows * sizeof(ushort), cudaMemcpyDeviceToHost);
	
	cudaFree(deviceA);
	cudaFree(deviceB);
	
	return B
}