#ifndef HAMC_REDUCE_H
#define HAMC_REDUCE_H

#include "hamc_cpu_code.c"


__global__ void shared_reverse_firstreduction (int * d_out, int * d_in){
    extern __shared__ float sdata[];
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    // load shared mem from global mem
    sdata [tid] = d_in[myId];
    __syncthreads();
    // make sure entire block is loaded!
    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)  {
            sdata[tid] += sdata[tid  + s];   }
        __syncthreads();
    }
    if (tid == 0){d_out[blockIdx.x] = sdata[tid]; }
}


int run_reduction_gpu(int *input, int len, int* version)
{
    // by default runs best performing kernel
    int kernel_to_run;
    if (!version) kernel_to_run = 0;
    else kernel_to_run = *version;

    int output;

    /* allocate device memory */
    int *deviceA;
    cudaMalloc((void **) &deviceA, len * sizeof(int));

    /* transfer host data to device */
    cudaMemcpy(deviceA, input, len * sizeof(int), cudaMemcpyHostToDevice);

    printf("Starting multiply matrix kernel...\n");

    // /* determine block and grid dimensions */
    //dim3 DimBlock(SCAN_TILE_WIDTH, SCAN_TILE_WIDTH, 1);
    //int x_blocks = ((A->rows - 1)/SCAN_TILE_WIDTH) + 1;
    //int y_blocks = ((A->cols - 1)/SCAN_TILE_WIDTH) + 1;
    //dim3 DimGrid(x_blocks, y_blocks, 1);

    //shared_reduce_reverse_first_reduction<<<blocks, threads, threads*sizeof(int)>>> ();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    //cudaMemcpy(C->data, deviceB, A->rows * A->cols * sizeof(HAMC_DATA_TYPE_t), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);

    return output;
}


#endif /* HAMC_SCAN_H */
