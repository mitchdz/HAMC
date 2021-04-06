#ifndef ENCRYPT_KERNEL_H
#define ENCRYPT_KERNEL_H

#include <stdio.h>
#include <time.h>


#include "TransposeMatrix.cu"
#include "MultiplyMatrix.cu"
#include "hamc_cpu_code.c"

#define TILE_WIDTH_MULTIPLY 16

#ifndef ushort
#define ushort unsigned short
#endif

bin_matrix run_matrix_multiply_kernel(bin_matrix A, bin_matrix B)
{
    bin_matrix C = mat_init_cpu(A->rows, B->cols);

    /* allocate device memory */
    ushort *deviceA;
    ushort *deviceB;
    ushort *deviceC;
    cudaMalloc((void **) &deviceA, A->rows * A->cols * sizeof(ushort));
    cudaMalloc((void **) &deviceB, B->rows * B->cols * sizeof(ushort));
    cudaMalloc((void **) &deviceC, C->rows * C->cols * sizeof(ushort));

    /* transfer host data to device */
    cudaMemcpy(deviceA, A->data, A->rows * A->cols * sizeof(ushort), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B->data, B->cols * B->rows * sizeof(ushort), cudaMemcpyHostToDevice);

    printf("Starting multiply matrix kernel...\n");

     /* determine block and grid dimensions */
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((A->rows - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((A->cols - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);

    mult_kernel<<<DimGrid, DimBlock>>> (deviceA, deviceB, deviceC, A->rows, B->rows, A->cols, B->cols);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(C->data, deviceC, C->rows * C->cols * sizeof(ushort), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return C;
}

bin_matrix generator_matrix_gpu(mdpc code)
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    //bin_matrix H = parity_check_matrix_cpu(code);

    //End of modified code
    printf("Construction of G started...\n");
    bin_matrix H_inv = circ_matrix_inverse_cpu(make_matrix_cpu(code->p, code->p,
               splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 1));

    //printf("H_inv generated...\n");
    //printf("stop\n");
    bin_matrix H_0 = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, 0,
               code->p), 1);

    //bin_matrix H_inv_times_H0 = mat_init_cpu(H_inv->rows, H_inv->cols);
    bin_matrix H_inv_times_H0 = run_matrix_multiply_kernel(H_inv, H_0);

    //TODO: transpose kernel
    // Tranpose H_inv*M
    bin_matrix Q = transpose_cpu(H_inv_times_H0);

    //printf("Transpose obtained...\n");
    bin_matrix M;

    int i;
    for(i = 1; i < code->n0 - 1; i++) {
        M = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, i * code->p, (i + 1) * code->p), 1);
        M = transpose_cpu(matrix_mult_cpu(H_inv, M));
        Q = concat_vertical_cpu(Q, M);
    }
    bin_matrix I = mat_init_cpu(code->k, code->k);
    make_indentity_cpu(I);
    bin_matrix G = concat_horizontal_cpu(I, Q);

    //bin_matrix G = mat_kernel(H);
    //G = matrix_rref(G);
    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("Time for G: %f\n", cpu_time_used);
    printf("Generator matrix generated....\n");
    return G;

}



//Initialize the mceliece cryptosystem
mcc mceliece_init_gpu(int n0, int p, int w, int t, unsigned seed)
{
    mcc crypt;
    crypt = (mcc)safe_malloc(sizeof(struct mceliece));
    crypt->code = qc_mdpc_init_cpu(n0, p, w, t, seed);
    crypt->public_key = generator_matrix_gpu(crypt->code);
    //printf("mceliece generated...\n");
    return crypt;
}

void test_gpu_e2e(int n0, int p, int t, int w, int seed)
{

    printf("GPU based mceliece cryptosystem test\n");

    printf("Starting Encryption...\n");
    clock_t start, end;
    double cpu_time_used;
    start = clock();


    mcc crypt = mceliece_init_gpu(n0, p, w, t, seed);


    bin_matrix msg = mat_init_cpu(1, crypt->code->k);
    //Initializing the message a random message
    for(int i = 0; i < crypt->code->k; i++)
    {
            int z = rand() % 2;
            set_matrix_element_cpu(msg, 0, i, z);
    }

    printf("message:\n");
    for (int i = 0; i < msg->cols; i++)
        printf("%hu", msg->data[i]);
    printf("\n");

    printf("public key:\n");
    for (int i = 0; i < crypt->public_key->cols; i++)
        printf("%hu", crypt->public_key->data[i]);
    printf("\n");

    bin_matrix error = get_error_vector_cpu(crypt->code->n, crypt->code->t);

    printf("error vector:\n");
    for (int i = 0; i < error->cols; i++)
        printf("%hu", error->data[i]);
    printf("\n");


    bin_matrix v = encrypt_cpu(msg, crypt);

    printf("encrypted data (message * public key + error):\n");
    for (int i = 0; i < v->cols; i++)
        printf("%hu", v->data[i]);
    printf("\n");


    bin_matrix s = decrypt_cpu(v, crypt);

    printf("decrypted data:\n");
    for (int i = 0; i < s->cols; i++)
        printf("%hu", s->data[i]);
    printf("\n");

    if(mat_is_equal_cpu(msg, s)) {
            end = clock();
            printf("Decryption successful...\n");
            cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
            printf("Time taken by cryptosystem: %f\n", cpu_time_used);
    } else {
            printf("Failure....\n");
    }
    delete_mceliece_cpu(crypt);
    return;
}







void run_encryption_gpu(const char* inputFileName, const char* outputFileName,
        int n, int p, int t, int w, int seed)
{
    //TODO:
}

#endif /* ENCRYPT_KERNEL_H */
