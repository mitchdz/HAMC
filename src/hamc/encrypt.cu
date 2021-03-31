#include <stdio.h>
#include <time.h>


#include "TransposeMatrix.cu"
#include "MultiplyMatrix.cu"


#define TILE_WIDTH_MULTIPLY 16

#ifndef ushort
#define ushort unsigned short
#endif



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

    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int x_blocks = ((code->p - 1)/TILE_WIDTH) + 1;
    int y_blocks = ((code->p - 1)/TILE_WIDTH) + 1;
    dim3 DimGrid(x_blocks, y_blocks, 1);

    bin_matrix Q = mat_init_cpu(code->p, code->p);
    bin_matrix H_inv_times_H0 = mat_init_cpu(code->p, code->p);

    ushort *deviceA;
    ushort *deviceB;
    ushort *deviceC;
    cudaMalloc((void **) &deviceA, code->p * code->p * sizeof(ushort));
    cudaMalloc((void **) &deviceB, code->p * code->p * sizeof(ushort));
    cudaMalloc((void **) &deviceC, code->p * code->p * sizeof(ushort));

    cudaMemcpy(deviceA, H_inv->data, code->p * code->p * sizeof(ushort), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, H_0->data, code->p * code->p * sizeof(ushort), cudaMemcpyHostToDevice);

    printf("Starting multiply matrix kernel...\n");
    // Multiple H_inv by matrix M
    mult_kernel<<<DimGrid, DimBlock>>>
        (deviceA, deviceB, deviceC,
         code->p, code->p, code->p, code->p);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(H_inv_times_H0->data, deviceC, code->p * code->p * sizeof(ushort), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    //H_inv_times_H0 = matrix_mult_cpu(H_inv, H_0);

    bool verbose = true;

    if (verbose) {
        printf("H0*H_inv:\n");
        for(int i = 0; i < code->p*2; i++) {
            for(int j = 0; j < code->p; j++) {
                printf("%hu", H_inv_times_H0->data[i*j + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    //transpose<<<DimGrid, DimBlock>>> (H_inv_times_H0->data, Q, code->p, code->p);

    //printf("Starting tranpose matrix kernel...\n");
    //transpose<<<DimGrid, DimBlock>>>
    //    (H_inv_times_M->data, M->data, code->p, code->p);

    // Tranpose H_inv*M
    Q = transpose_cpu(H_inv_times_H0);




    //Q = transpose_cpu(matrix_mult_cpu(H_inv,  H_0));


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
//    long f_size;
//    int c;
//    size_t icc = 0;
//
//    /* open input and output files */
//    FILE *in_file  = fopen(inputFileName, "r");
//    FILE *out_file  = fopen(outputFileName, "w");
//
//    // test for input file not existing
//    if (in_file == NULL) {
//        printf("Error! Could not open file\n");
//        exit(-1);
//    }
//
//    /* determine filesize */
//    fseek(in_file, 0, SEEK_END);
//    f_size = ftell(in_file);
//    fseek(in_file, 0, SEEK_SET);
//
//    char *input_message = (char *)malloc(f_size);
//
//    /* read and write file data into local array */
//    icc = 0;
//    while ((c = fgetc(in_file)) != EOF) {
//        input_message[icc++] = (char)c;
//    }
//    input_message[icc] = '\0';
//
//
//    printf("\n");
//    printf("Input message (char):\n");
//    for (int i = 0; i < (int)icc; i++)
//        printf("%c",  input_message[i]);
//    printf("\n");
//
//    printf("Input message (ushort):\n");
//    for (int i = 0; i < (int)icc; i++)
//        printf("%hu ",  (ushort)input_message[i]);
//    printf("\n");
//
//
//    printf("\n");
//
//    /* check that input file is within size */
//    int k = (n - 1) * p;
//    if ((int)icc > k) {
//        printf("ERROR: intput message is too long for k\n");
//        printf("input message is length %d while k is %d\n", (int)icc, k);
//    }
//
//
//    /* set up basic encryption primitives */
//    crypt = mceliece_init_cpu(n, p, w, t, seed);
//    msg = mat_init_cpu(1, k);
//    for(int i = 0; i < k; i++) {
//        set_matrix_element_cpu(msg, 0, i,
//                (unsigned short)strtoul(input_message, NULL, 0));
//    }
//
//    printf("\n");
//    /* run CPU based encryption code */
//    m = encrypt_cpu(msg, crypt);
//    printf("Encrypted data (ushort):\n");
//    for (int i = 0; i < m->cols; i++)
//        printf("%hu", m->data[i]);
//    printf("\n");
//
//    /* decrypt the ciphertext */
//    bin_matrix d = decrypt_cpu(m, crypt);
//
//    printf("Decrypted text:\n");
//    for(int i = 0; i < d->cols; i++)
//        printf("%c", (char)d->data[i]);
//
//
//    /* write cipher.text to file */
//    for( int i = 0; i < m->cols; i++) {
//        fprintf(out_file, "%hu", get_matrix_element_cpu(m, 0, i));
//    }
//
//cleanup:
//    fclose(in_file);
//    fclose(out_file);
}