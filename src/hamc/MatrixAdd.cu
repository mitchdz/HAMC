//#include <wb.h>
//
//#define HAMC_DATA_TYPE_t HAMC_DATA_TYPE_t
//
//__global__ void MatrixAdd(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C,
//                                     int height, int width,) {
//        int ROW = blockIdx.y*blockDim.y + threadIdx.y;
//	
//	int COL = blockIdx.x*blockDim.x + threadIdx.x;
//	
//	if((ROW < height) && (COL < width)){
//		int address = ROW*width+COL;
//		C[address] = A[i] ^ B[i];
//	}
//}
