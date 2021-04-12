
#ifndef TEST_INVERSE_CPU_H
#define TEST_INVERSE_CPU_H

#include "../hamc/hamc_cpu_code.c"
#include "../hamc/InverseMatrix.cu"
#include "debug_inverse_c.c"

void print_bin_matrix(bin_matrix A);
bin_matrix my_circ_matrix_inverse_cpu(bin_matrix A);

int main()
{
    int seed = 10;

    printf("\n");
    printf("Inverse test\n");

    clock_t cpu_start, cpu_end;
    double cpu_time_used;

    clock_t gpu_start, gpu_end;
    double gpu_time_used;


    int rows = 4;
    int cols = 4;

    srand(seed);

    bin_matrix msg_cpu = mat_init_cpu(rows, cols);
    bin_matrix msg_gpu = mat_init_cpu(rows, cols);
    //Initializing the message a random message
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int z = rand() % 2;
            set_matrix_element_cpu(msg_cpu, i, j, z);
            set_matrix_element_cpu(msg_gpu, i, j, z);
        }
    }

    // print random input data:
    printf("input matrix %dx%d:\n", rows, cols);
    print_bin_matrix(msg_cpu);


    cpu_start = clock();

    bin_matrix inverted_cpu = my_circ_matrix_inverse_cpu(msg_cpu);

    cpu_end = clock();
    cpu_time_used = ((double) (cpu_end - cpu_start))/ CLOCKS_PER_SEC;



    gpu_start = clock();

    bin_matrix inverted_gpu = run_inverse_kernel(msg_gpu);

    gpu_end = clock();
    gpu_time_used = ((double) (gpu_end - gpu_start))/ CLOCKS_PER_SEC;


    printf("Inverted matrix from CPU:\n");
    print_bin_matrix(inverted_cpu);
    printf("Inverted matrix from GPU:\n");
    print_bin_matrix(inverted_gpu);

    printf("Time for CPU: %f\n", cpu_time_used);
    printf("Time for GPU: %f\n", gpu_time_used);


    bool correct = true;
    for (int i =0; i < msg_cpu->rows*msg_cpu->cols; i++) {
        if (inverted_cpu->data[i] != inverted_gpu->data[i]) {
            correct = false;
            break;
        }
    }

    printf("correctq: ");
    if (correct) printf("true\n");
    else (printf("false\n"));

    free(msg_cpu);
    free(msg_gpu);

    free(inverted_gpu);
    free(inverted_cpu);

    return 0;
}

void print_bin_matrix(bin_matrix A)
{
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            printf("%hu ", A->data[i*A->cols + j]);
        }
        printf("\n");
    }
}


#endif /* TEST_INVERSE_CPU_H */
