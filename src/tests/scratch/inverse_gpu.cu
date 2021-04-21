

#ifndef INVERSE_GPU_H
#define INVERSE_GPU_H

#define BLOCK_SIZE_LU 16
#define BLOCK_SIZE_LU2 256

__global__ void ForwardSolve(HAMC_DATA_TYPE_t* A, HAMC_DATA_TYPE_t* B, int n, int k, int half_k, int i){
    int ty = threadIdx.y;
    int by = blockIdx.y;
    int tidy = by*BLOCK_SIZE_LU2+ty;
    int row = tidy + i + 1;
    __shared__ HAMC_DATA_TYPE_t mult;

    if(ty==0){
        mult = B[i];
    }

    __syncthreads();

    if(tidy < half_k && row < n){
        B[row] ^= A[row*k + half_k - 1 - tidy] & mult;
    }
}



__global__ void BackSolve(HAMC_DATA_TYPE_t* A, HAMC_DATA_TYPE_t* B, int rows, int k, int half_k, int i){
    int ty = threadIdx.y;
    int by = blockIdx.y;

    int tidy = by*BLOCK_SIZE_LU2+ty;
    int row = i - 1 - tidy;
    __shared__ HAMC_DATA_TYPE_t mult;

    if(ty==0){
        mult = B[i];
    }

    __syncthreads();

    if(tidy < half_k && row >= 0){
        B[row] ^=  A[row*k + half_k + 1 + tidy] & mult;
    }
}


__global__ void reduce(HAMC_DATA_TYPE_t* A, int rows, int cols, int i){

  int tx = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = blockIdx.y*blockDim.x + threadIdx.y;

  int row = ty + i + 1;
  int col = tx + ty;

  if(row<rows && col < cols){
    A[row*cols + col] ^= A[row*cols + col - tx - 1] & A[i*cols + tx + 1];
  }

}

// A is the input matrix, this will be modified. It is supposed to be modified.
// P is the pivot vector
// size is the dimension of the square matrix
// index is the column that we are operating on
__global__ void reduce_GF2(HAMC_DATA_TYPE_t* A, HAMC_DATA_TYPE_t* P, int size, int index)
{
    extern __shared__ HAMC_DATA_TYPE_t pivot[];

    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.x + threadIdx.y;

    // at start, load P into pivot
    if (tx == 0) {
        for (int i = 0; i < size; i++) {
            pivot[i] = P[i];
        }
    }


    if (tx == 0) {
        for ( int i = index; i < size; i++) {
            pivot[i] = A[(index*size)+i];
        }
    }

    __syncthreads();

    int pivot_row = index * size;
    int my_row = tx * size;

    int start = my_row + index;
    int end = my_row + size;

    if (my_row > pivot_row) {
        for (int i = start + 1; i < end; i++) {
            A[i] ^= A[start] & pivot[(i-my_row)];
        }
    }

    //TODO: have each thread handle a different index
    // write every value into P
    if (tx == 0) {
        for (int i = 0; i < size; i++) {
            P[i] = pivot[i];
        }
    }


}

// P = [1,2,3,4...,N]


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Row elimination Kernel - takes matrix, dimension, currect row index, and block size
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void elim_GF2_kernel(HAMC_DATA_TYPE_t *A, int n, int index, int bsize){
	extern __shared__ HAMC_DATA_TYPE_t pivot[];

	int idThread=threadIdx.x;
	int idBlock=blockIdx.x;
	int blockSize=bsize;

    // fill pivot for this kernel
	if(idThread==0){
	     for(int i=index;i<n;i++) pivot[i]=A[(index*n)+i];
	}

    // wait for pivot vector to be populated
	__syncthreads();


    //Varitables for pivot, row, start and end
	int pivotRow=(index*n);
	int currentRow=(((blockSize*idBlock) + idThread)*n);
	int start=currentRow+index;
	int end=currentRow+n;

    //If greater than pivot row, loop from start index + 1(next row) to end of column
	if(currentRow > pivotRow){
        for(int i= start+1; i<end; ++i){
            //Set the matrix value of next row and its column - pivot
            //A[i]=A[i]-(A[start]*pivot[i-currentRow]);
            A[i] ^= A[start] & pivot[i-currentRow];
        }
        // swap current row w/ P row
        //HAMC_DATA_TYPE_t temp = P[pivotRow];
        //P[pivotRow] = P[currentRow];
        //P[currentRow] = temp;
    }
}

// 1) A = P * L * U
// 2) y*U = I // y is unkown
// 3) z*L = y // z is unkown
// 4) x*P = z // x is unkown, x is the inverse of A
bin_matrix inverse_GF2_gpu(bin_matrix A)
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
    HAMC_DATA_TYPE_t *d_B = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t)*A->rows*A->cols);
    HAMC_DATA_TYPE_t *d_P = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t)*A->cols);
    HAMC_DATA_TYPE_t *h_P = (HAMC_DATA_TYPE_t *)malloc(sizeof(HAMC_DATA_TYPE_t)*A->cols);

    for (int i = 0; i < A->cols; i++) {
        h_P[i] = i;
    }


    //allocate GPU memory
    cudaMalloc ( (void**)&d_A, A->rows*A->cols*sizeof(HAMC_DATA_TYPE_t) );
    cudaMalloc ( (void**)&d_B, A->rows*A->cols*sizeof(HAMC_DATA_TYPE_t) );
    cudaMalloc ( (void**)&d_P, A->cols*sizeof(HAMC_DATA_TYPE_t) );

    // Copy bin_matrix A data to GPU
    cudaMemcpy(d_A, A->data,
        A->rows*A->cols*sizeof(HAMC_DATA_TYPE_t),
        cudaMemcpyHostToDevice);


    // Initialize P
    cudaMemcpy(d_P, h_P,
        A->cols*sizeof(HAMC_DATA_TYPE_t),
        cudaMemcpyHostToDevice);



    if (verbose) printf("\nPerforming LU Decomposition...\n");

    int k = 31;
    int half_k = (k-1)/2;


    int tile = 16;
    int numblock = A->cols/tile + ((A->cols%tile)?1:0);

    /* LU decomposition */
    LU_start = clock();
    for(int i = 0; i < A->cols; i++){
        //int blocks=((A->cols/512));
        //reduce_GF2<<<blocks,512,A->cols*sizeof(HAMC_DATA_TYPE_t)>>>(d_A,d_P,A->cols,i);
        elim_GF2_kernel<<<numblock,tile,A->cols*sizeof(HAMC_DATA_TYPE_t)>>>(d_A,A->cols,i,tile);
    }
    LU_end = clock();
    LU_time_used = ((double) (LU_end - LU_start)) / CLOCKS_PER_SEC;


    //cudaMemcpy( h_P, d_P, A->cols*sizeof(HAMC_DATA_TYPE_t),  cudaMemcpyDeviceToHost );

    //printf("P Vector:\n");
    //for (int i = 0; i < A->cols; i++)printf("%d ", h_P[i]);
    //printf("\n");



    printf("\nOutput matrix:\n");
    cudaMemcpy( output_matrix->data, d_A,
        A->rows*A->cols*sizeof(HAMC_DATA_TYPE_t),
        cudaMemcpyDeviceToHost );

    printf("\n");
    print_bin_matrix(output_matrix);
    printf("\n");
    

    /* INPUT: A,P filled in LUPDecompose; N - dimension
    * OUTPUT: IA is the inverse of the initial matrix
    */
    /*
    void LUPInvert(double **A, int *P, int N, double **IA) {
    
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                IA[i][j] = P[i] == j ? 1.0 : 0.0;

                for (int k = 0; k < i; k++)
                    IA[i][j] -= A[i][k] * IA[k][j];
            }

            for (int i = N - 1; i >= 0; i--) {
                for (int k = i + 1; k < N; k++)
                    IA[i][j] -= A[i][k] * IA[k][j];

                IA[i][j] /= A[i][i];
            }
        }
    }
    */

    bin_matrix IA = mat_init_cpu(A->rows, A->cols);
    

    h_P[0] = 1;

    clock_t fb_start = clock();
    if (verbose) printf("\nPerforming Forward backward substition...\n");

    // Forward backward substitution in C
    for (int j = 0; j < A->rows; j++) {
        for (int i = 0; i < A->cols; i++) {
            IA->data[i*A->cols + j] = h_P[i] == j ? 1.0 : 0.0;

            for (int k = 0; k < i; k++)
                IA->data[i*A->cols + j] ^= A->data[i*A->cols + k] & IA->data[k*A->cols + j];
        }

        for (int i = A->cols - 1; i >= 0; i--) {
            for (int k = i + 1; k < A->cols; k++)
                IA->data[i*A->cols + j] ^= A->data[i*A->cols + k] & IA->data[k*A->cols + j];

            //IA[i][j] /= A[i][i];
        }
    }


    printf("\nAfter C based BF\n");
    print_bin_matrix(IA);

    free(IA);

    /* Forward Backward Substitution */
    /*
    for ( int i = 0; i < A->cols; i++) {
        cudaMemcpyAsync(d_B, h_B+1*A->cols, sizeof(HAMC_DATA_TYPE_t)*A->cols, cudaMemcpyHostToDevice);

        for (int j = 0; j < A->cols; j++) {
            ForwardSolve<<<dimGrid3, dimBlock3>>>(d_A, d_B, A->cols, k, half_k, j);
        }
        for (int j = A->cols - 1; j >= 0; j--) {
            BackSolve<<<dimGrid3, dimBlock3>>>(d_A, d_B, A->cols, k, half_k, j);
        }

        cudaMemcpyAsync(h_B+1*A->cols, d_B, sizeof(HAMC_DATA_TYPE_t)*A->cols, cudaMemcpyDeviceToHost);
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
        print_bin_matrix(IA);
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


#endif /* INVERSE_GPU_H */
