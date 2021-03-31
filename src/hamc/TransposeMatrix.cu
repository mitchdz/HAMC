#define ushort unsigned short

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
