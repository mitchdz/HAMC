#ifndef HAMC_SCRATCH_H
#define HAMC_SCRATCH_H


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>

#include <sys/time.h>


#include "../../src/hamc/hamc_cpu_code.c"

using namespace std;


__global__ void add(HAMC_DATA_TYPE_t *A, HAMC_DATA_TYPE_t *B,
        HAMC_DATA_TYPE_t *C)
{
    int tid = blockIdx.x;
    //C[tid] = A[tid] + B[tid];
    C[tid] = A[tid] | B[tid];
}


__global__ void scale(HAMC_DATA_TYPE_t *A, int size, int index)
{
    int i;
    int start = index*size + index;
    int end = index*size+size;

    for (i = start+1; i < end;i++) {
        //A[i]=(A[i]/A[start]);
        //A[i]=(A[i]^A[start]);
    }
}

__global__ void reduce(HAMC_DATA_TYPE_t *A, int size, int index, int b_size)
{
    extern __shared__ HAMC_DATA_TYPE_t pivot[];



    int i;

    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int block_size=b_size;

    int pivot_start=(index*size+index);
    int pivot_end=(index*size+size);

    int start;
    int end;
    int pivot_row;
    int my_row;

    if(tid==0){
        for(i=index;i<size;i++) pivot[i]=A[(index*size)+i];
    }

    __syncthreads();

    pivot_row=(index*size);
    my_row=(((block_size*bid) + tid)*size);
    start=my_row+index;
    end=my_row+size;

    if(my_row >pivot_row){
        for(i=start+1;i<end;i++){
            //A[i]=A[i]-(A[start]*pivot[(i-my_row)]);
            A[i] ^= (A[start] & pivot[(i-my_row)]);
        }
    }



}


int main(int argc, char *argv[]){
    printf("Scratch test\n");
    HAMC_DATA_TYPE_t *a;
    HAMC_DATA_TYPE_t *c;
    int N;
    int flag=0;

    HAMC_DATA_TYPE_t **result;
    HAMC_DATA_TYPE_t **b;
    int blocks;

    HAMC_DATA_TYPE_t *dev_a;
    int i;
    int j;

    double start;
    double end;
    struct timeval tv;

    //N=atoi(argv[1]);

    N = 8;

    //allocate memory on CPU
    a=(HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t)*N*N);
    c=(HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t)*N*N);


    result=(HAMC_DATA_TYPE_t **)malloc(sizeof(HAMC_DATA_TYPE_t *)*N);
    b=     (HAMC_DATA_TYPE_t **)malloc(sizeof(HAMC_DATA_TYPE_t *)*N);


    for(i=0;i<N;i++){
       result[i]=(HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t)*N);
       b[i]     =(HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t)*N);
    }

    //allocate the memory on the GPU
    cudaMalloc ( (void**)&dev_a, N*N* sizeof (HAMC_DATA_TYPE_t) );


    bin_matrix sol_raw = mat_init_cpu(N,N);

    srand((unsigned)2);
    //fill the arrays 'a' on the CPU
    for ( i = 0; i <= (N*N); i++) {
        HAMC_DATA_TYPE_t val = ((rand()%2));
        a[i] = val;
        sol_raw->data[i] = val;
    }

    printf("Matrix a is :\n");
    for(i=0; i<(N*N); i++){
        if(i%N==0)
            printf("\n %d ", a[i]);
        else
            printf("%d ",a[i]);
    }
    printf("\n\n");

    bin_matrix sol = circ_matrix_inverse_cpu(sol_raw);

    printf("Expected solution is :\n");
    for(i=0; i<(N*N); i++){
        if(i%N==0)
            printf("\n %d ", sol->data[i]);
        else
            printf("%d ",sol->data[i]);
    }




    cudaMemcpy(dev_a,a,N*N*sizeof(HAMC_DATA_TYPE_t), cudaMemcpyHostToDevice);//copy array to device memory

    gettimeofday(&tv,NULL);
    start=tv.tv_sec;
    /*Perform LU Decomposition*/
    printf("\n==========================================================\n"); 
    for(i=0;i<N;i++){
        //scale<<<1,1>>>(dev_a,N,i);
        // blocks= ((N-i-1)/512)+1;
        blocks=((N/512));
        //printf("Number of blocks rxd : %d \n",blocks);
        reduce<<<blocks,512,N*sizeof(HAMC_DATA_TYPE_t)>>>(dev_a,N,i,512);

    }
    /*LU decomposition ends here*/

    gettimeofday(&tv,NULL);
    end=tv.tv_sec;
    cudaMemcpy( c, dev_a, N*N*sizeof(HAMC_DATA_TYPE_t),cudaMemcpyDeviceToHost );//copy array back to host

    printf("\nThe time for LU decomposition is %lf \n",(end-start));
       //display the results
    printf("\nOutput from GPU is \n");
    for ( i = 0; i < (N*N); i++) {
               if(i%N==0)
        printf( "\n%d  ", c[i]);
               else  printf("%d ",c[i]);
    }
    printf("\n");

    printf("=======================================================");
    printf("\n Performing inplace verification \n");
        /*Inplace verification step*/

    for (int i = 0; i < N*N; i++) {
        if (sol->data[i] != c[i]) {
            flag = 1;
            break;
        }
    }


    if(flag==0) printf("correctq: Correct");
    else printf("correctq: Failure %d \n",flag);

    //free the memory allocated on the GPU
    cudaFree( dev_a );

    return 0;
}



#endif /* HAMC_SCRATCH_H */
