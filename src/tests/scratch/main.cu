#ifndef HAMC_SCRATCH_H
#define HAMC_SCRATCH_H


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>


#include "../../src/hamc/hamc_cpu_code.c"

using namespace std;


#define blocksize 8

__global__ void nodiag_normalize(uint8_t *A, uint8_t *I, int n, int i){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        if (x == i && x!=y) {
            I[x*n + y] ^= A[i*n + i];
            A[x*n + y] ^= A[i*n + i];
        }
    }
}

__global__ void diag_normalize(uint8_t *A, uint8_t *I, int n, int i){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        if (x == y && x == i){
            I[x*n + y] ^= A[i*n + i];
            A[x*n + y] ^= A[i*n + i];
        }
    }

}

__global__ void gaussjordan(uint8_t *A, uint8_t *I, int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n){
        if (x != i){
            //I[x*n + y] -= I[i*n + y] * A[x*n + i];
            I[x*n + y] ^= I[i*n + y] & A[x*n + i];
            if (y != i){
                //A[x*n + y] -= A[i*n + y] * A[x*n + i];
                A[x*n + y] ^= A[i*n + y] & A[x*n + i];
            }
        }
    }

}

__global__ void set_zero(uint8_t *A, uint8_t *I, int n, int i){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n){
        if (x != i){
            if (y == i){
                A[x*n + y] = 0;
            }
        }
    }
}


void printMatrix(uint8_t *mat, int n)
{
    for ( int i = 0; i < n; i++) {
        for ( int j = 0; j < n; j++) {
            printf("%d  ", mat[i*n+j]);
        }
        printf("\n");
    }
}




int main()
{
    const int n = 4;
    // creating input
    uint8_t *iL = new uint8_t[n*n];
    uint8_t *L = new uint8_t[n*n];

    bin_matrix CPU = mat_init_cpu(n,n);

    uint8_t val;
    int seed = 10;
    // create random nxn binary matrix
    srand(seed);
    for ( int i = 0; i < n*2; i++) {
        val = rand() %2;
        L[i] = val;
        CPU->data[i] = val;
    }

    printMatrix(L,n);
    printf("\n");

    //savetofile(L, "L.txt", n, n);

    cout << "inv\n";
    uint8_t *d_A, *I, *dI;
    float time;
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int ddsize = n*n*sizeof(uint8_t);

    dim3 threadsPerBlock(blocksize, blocksize);
    dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);


    // memory allocation
    err = cudaMalloc((void**)&d_A, ddsize);
    if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    err = cudaMalloc((void**)&dI, ddsize);
    if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    I = new uint8_t[n*n];

    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            if (i == j) I[i*n + i] = 1.0;
            else I[i*n + j] = 0.0;
        }
    }

    //copy data from CPU to GPU
    err = cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    err = cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }

    //timer start
    cudaEventRecord(start, 0);

    // L^(-1)
    for (int i = 0; i<n; i++){
        nodiag_normalize <<<numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
        diag_normalize <<<numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
        gaussjordan <<<numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
        set_zero <<<numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //copy data from GPU to CPU
    err = cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    err = cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }

    cout << "Cuda Time - inverse: " << time << "ms\n";
    //savetofile(I, "I.txt", n, n);
    cudaFree(d_A);
    cudaFree(dI);

    printf("iL:\n");
    printMatrix(iL, n);
    printf("\n");


    bin_matrix out = circ_matrix_inverse_cpu(CPU);

    free(CPU);
    free(out);

    printf("CPU output:\n");
    printMatrix(out->data, n);
    printf("\n");



    uint8_t *c = new uint8_t[n*n];
    for (int i = 0; i<n; i++) {
        for (int j = 0; j<n; j++) {
            c[i*n+j] = 0;  //put the initial value to zero
            for (int x = 0; x<n; x++)
                //c[i*n + j] = c[i*n + j] + L[i*n+x] * iL[x*n + j];  //matrix multiplication
                c[i*n + j] = c[i*n + j] + L[i*n+x] & iL[x*n + j];  //matrix multiplication
        }
    }

    printf("output:\n");
    printMatrix(c, n);
    printf("\n");



    return 0;
}




#endif /* HAMC_SCRATCH_H */
