
#ifndef TRANSPOSE_KERNEL_H
#define TRANSPOSE_KERNEL_H

#define BLOCK_DIM 16
#define TRANSPOSE_TILE_WIDTH 16
#include "hamc_cpu_code.c"

__global__ void transpose_no_bank_conflicts(HAMC_DATA_TYPE_t *idata, HAMC_DATA_TYPE_t *odata, int width, int height);
__global__ void transpose_naive(HAMC_DATA_TYPE_t *idata, HAMC_DATA_TYPE_t* odata, int width, int height);

bin_matrix run_transpose_kernel(bin_matrix A)
{
    /* transpose so rows/cols flipped */
    bin_matrix C = mat_init_cpu(A->cols, A->rows);

    /* allocate device memory */
    HAMC_DATA_TYPE_t *deviceA;
    HAMC_DATA_TYPE_t *deviceB;
    cudaMalloc((void **) &deviceA, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));
    cudaMalloc((void **) &deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t));

    /* transfer host data to device */
    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);

    printf("Starting Transpose matrix kernel...\n");

     /* determine block and grid dimensions */
    dim3 DimBlock(TRANSPOSE_TILE_WIDTH, TRANSPOSE_TILE_WIDTH, 1);
    int x_blocks = ((A->rows - 1)/TRANSPOSE_TILE_WIDTH) + 1;
    int y_blocks = ((A->cols - 1)/TRANSPOSE_TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);

    transpose_no_bank_conflicts<<<DimGrid, DimBlock>>> (deviceA, deviceB, A->rows, A->cols);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(C->data, deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);

    return C;
}

__global__ void transpose_no_bank_conflicts(HAMC_DATA_TYPE_t *idata, HAMC_DATA_TYPE_t *odata, int width, int height)
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


__global__ void transpose_naive(HAMC_DATA_TYPE_t *idata, HAMC_DATA_TYPE_t* odata, int width, int height)
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
