#ifndef HAMC_SCRATCH_H
#define HAMC_SCRATCH_H


#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>

#include <sys/time.h>


#include "../../src/hamc/hamc_cpu_code.c"

using namespace std;

#define BLOCK_SIZE_LU 16
#define BLOCK_SIZE_LU2 256

__global__ void ForwardSolve(HAMC_DATA_TYPE_t* A, HAMC_DATA_TYPE_t* b, int n, int k, int half_k, int i){
    int ty = threadIdx.y;
    int by = blockIdx.y;
    int tidy = by*BLOCK_SIZE_LU2+ty;
    int row = tidy + i + 1;
    __shared__ HAMC_DATA_TYPE_t mult;

    if(ty==0){
        mult = b[i];
    }

    __syncthreads();

    if(tidy < half_k && row < n){
        b[row] ^= A[row*k + half_k - 1 - tidy] & mult;
    }
}



__global__ void BackSolve(HAMC_DATA_TYPE_t* A, HAMC_DATA_TYPE_t* b, int n, int k, int half_k, int i){
  int ty = threadIdx.y;
  int by = blockIdx.y;
  int tidy = by*BLOCK_SIZE_LU2+ty;
  int row = i - 1 - tidy;
  __shared__ HAMC_DATA_TYPE_t mult;

  if(ty==0){
    b[i] = b[i]/A[i*k + half_k];
    mult = b[i];
  }

  __syncthreads();

  if(tidy < half_k && row >= 0){
    b[row] ^=  A[row*k + half_k + 1 + tidy] & mult;
  }

}


__global__ void add(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B, HAMC_DATA_TYPE_t *C)
{
    int tid = blockIdx.x;

    C[tid] = A[tid] ^ B[tid];
}

__global__ void reduce(HAMC_DATA_TYPE_t *A, int size, int index, int b_size)
{
    extern __shared__ HAMC_DATA_TYPE_t pivot[];

    int i;

    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int block_size=b_size;

    //int pivot_start=(index*size+index);
    //int pivot_end=(index*size+size);

    int start, end, pivot_row, my_row;

    if(tid==0){
        for(i=index;i<size;i++)  {
            pivot[i]=A[(index*size)+i];
        }
    }

    __syncthreads();

    pivot_row=(index * size);
    my_row=(((block_size * bid) + tid) * size);
    start=my_row + index;
    end=my_row + size;

    if(my_row >pivot_row){
        for(i=start+1 ; i < end;i++){
            //A[i]=A[i]-(A[start]*pivot[(i-my_row)]);
            A[i] ^= (A[start] & pivot[(i-my_row)]);
        }
    }
}


void print_bin_matrix(bin_matrix A)
{
    printf(" ");
    for ( int i = 0; i < A->rows; i++) {
        for ( int j = 0; j < A->cols; j++) {
            printf("%d ", A->data[i*A->cols + j]);
        }
        printf("\n ");
    }
}


// 1) A = P * L * U
// 2) y*U = I // y is unkown
// 3) z*L = y // z is unkown
// 4) x*P = z // x is unkown, x is the inverse of A
bin_matrix inverse_gpu(bin_matrix A)
{
    bool verbose = true;

    clock_t total_start, total_end, 
            LU_start, LU_end;

    double LU_time_used,
           total_time_used;


    total_start = clock();

    bin_matrix output_matrix = mat_init_cpu(A->rows, A->cols);


    //allocate CPU memory
    HAMC_DATA_TYPE_t *d_A = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t)*A->rows*A->cols);
    HAMC_DATA_TYPE_t *d_b = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t)*A->rows*A->cols);

    //allocate GPU memory
    cudaMalloc ( (void**)&d_A, A->rows*A->cols*sizeof(HAMC_DATA_TYPE_t) );

    // Copy bin_matrix A data to GPU
    cudaMemcpy(d_A, A->data,
        A->rows*A->cols*sizeof(HAMC_DATA_TYPE_t),
        cudaMemcpyHostToDevice);

    if (verbose) printf("\nPerforming LU Decomposition...\n");

    /* LU decomposition */
    LU_start = clock();
    for(int i = 0; i < A->cols; i++){
        int blocks=((A->cols/512));
        reduce<<<blocks,512,A->cols*sizeof(HAMC_DATA_TYPE_t)>>>
            (d_A,A->cols,i,512);
    }
    LU_end = clock();
    LU_time_used = ((double) (LU_end - LU_start)) / CLOCKS_PER_SEC;



    if (verbose) printf("\nPerforming Forward backward substition...\n");


    clock_t fb_start = clock();

    /* Forward Backward Substitution */
    /*
    for ( int i = 0; i < A->cols; i++) {
        for (int j = 0; j < A->cols; j++) {
            // Forward solve
        }
        for (int j = A->cols - 1; j >= 0; j++) {
            // Backwards solve
        }
    }
    */

    clock_t fb_end = clock();
    double fb_time_used = ((double) (fb_end - fb_start)) / CLOCKS_PER_SEC;


    cudaMemcpy( output_matrix->data, d_A,
        A->rows*A->cols*sizeof(HAMC_DATA_TYPE_t),
        cudaMemcpyDeviceToHost );

    total_end = clock();
    total_time_used = ((double) (total_end - total_start)) / CLOCKS_PER_SEC;

    if (verbose) {
        printf("\nfinal result:\n");
        print_bin_matrix(output_matrix);
    }


    if (verbose) {
        printf("\ntotal time used: %.2lfs\n", total_time_used);
        printf("LU Decomposition time used %.2lfs - %.2lf%%\n", 
            LU_time_used, 100*(LU_time_used/total_time_used));

        printf("Forward Backward substitution time used %.2lfs - %.2lf%%\n",
            fb_time_used, 100*(fb_time_used/total_time_used));
    }


    cudaFree(d_A);

    return output_matrix;
}





int main(int argc, char *argv[]){

    bool verbose = true;

    printf("Scratch test\n");
    int N;
    int flag=0;

    int i;


    N = 3;

    bin_matrix cpu_raw = mat_init_cpu(N,N);
    bin_matrix gpu_raw = mat_init_cpu(N,N);

    srand((unsigned)2);
    //fill the arrays 'a' on the CPU
    for ( i = 0; i <= (N*N); i++) {
        HAMC_DATA_TYPE_t val = ((rand()%2));
        cpu_raw->data[i] = val;
        gpu_raw->data[i] = val;
    }

    bin_matrix cpu_sol = circ_matrix_inverse_cpu(cpu_raw);

    if (verbose) {
        printf("\nInput matrix:\n");
        print_bin_matrix(gpu_raw);

        printf("\nExpected solution is:\n");
        print_bin_matrix(cpu_sol);
    }

    bin_matrix gpu_sol = inverse_gpu(gpu_raw);

    // check results
    for (int i = 0; i < N*N; i++) {
        if (gpu_sol->data[i] != cpu_sol->data[i]) {
            flag = 1;
            break;
        }
    }

    if(flag==0) printf("correctq: Correct");
    else printf("correctq: Failure\n");


    free(cpu_raw);
    free(gpu_raw);
    free(cpu_sol);
    free(gpu_sol);

    return 0;
}


#endif /* HAMC_SCRATCH_H */