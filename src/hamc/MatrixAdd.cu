#include <wb.h>

#define ushort unsigned short

__global__ void MatrixAdd(ushort *A, ushort *B, ushort *C,
                                     int height, int width,)
    int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	
    int COL = blockIdx.x*blockDim.x + threadIdx.x;
	
    int index = row * N + col;
 
    if( (ROW < numARows) && (COL < numAColumns) )
    {
  	C[index] = A[index] ^ B[index];
  	
    }
}
