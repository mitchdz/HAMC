
#ifndef TRANSPOSE_KERNEL_H
#define TRANSPOSE_KERNEL_H

#define BLOCK_DIM 16
#define TRANSPOSE_TILE_WIDTH 16
#include "hamc_cpu_code.c"

__global__ void transpose_no_bank_conflicts(ushort *idata, ushort *odata, int width, int height);
__global__ void transpose_naive(ushort *idata, ushort* odata, int width, int height);

bin_matrix run_transpose_kernel(bin_matrix A)
{
    /* transpose so rows/cols flipped */
    bin_matrix C = mat_init_cpu(A->cols, A->rows);

    /* allocate device memory */
    ushort *deviceA;
    ushort *deviceB;
    cudaMalloc((void **) &deviceA, A->rows * A->cols * sizeof(ushort));
    cudaMalloc((void **) &deviceB, A->rows * A->cols * sizeof(ushort));

    /* transfer host data to device */
    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(ushort), cudaMemcpyHostToDevice);

    printf("Starting multiply matrix kernel...\n");

     /* determine block and grid dimensions */
    dim3 DimBlock(TRANSPOSE_TILE_WIDTH, TRANSPOSE_TILE_WIDTH, 1);
    int x_blocks = ((A->rows - 1)/TRANSPOSE_TILE_WIDTH) + 1;
    int y_blocks = ((A->cols - 1)/TRANSPOSE_TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);

    transpose_no_bank_conflicts<<<DimGrid, DimBlock>>> (deviceA, deviceB, A->rows, A->cols);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(C->data, deviceB, A->rows * A->cols * sizeof(ushort), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);

    return C;
}

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory
// so that bank conflicts do not occur when threads address the array column-wise.
__global__ void transpose_no_bank_conflicts(ushort *idata, ushort *odata, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

    // read the matrix tile into shared memory
    int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}


// This naive transpose kernel suffers from completely non-coalesced writes.
// It can be up to 10x slower than the kernel above for large matrices.
__global__ void transpose_naive(ushort *idata, ushort* odata, int width, int height)
{
   int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

   if (xIndex < width && yIndex < height)
   {
       unsigned int index_in  = xIndex + width * yIndex;
       unsigned int index_out = yIndex + height * xIndex;
       odata[index_out] = idata[index_in];
   }
}

#endif // TRANSPOSE_KERNEL_H
