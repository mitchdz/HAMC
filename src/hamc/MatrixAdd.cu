#include <wb.h>

#define ushort unsigned short

__global__ void MatrixAdd(ushort *A, ushort *B, ushort *C,
                                     int height, int width,) {
        int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	
	int COL = blockIdx.x*blockDim.x + threadIdx.x;
	
	if((ROW < height) && (COL < width)){
		int address = ROW*width+COL;
		C[address] = A[i] ^ B[i];
	}
}
