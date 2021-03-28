#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "matrix.h"
#include "MultiplyMatrix.h"
#include "qc_mdpc.h"

//Rotate the row x positions to the right
unsigned short* shift_cpu(unsigned short* row, int x, int len)
{
    unsigned short* temp = (unsigned short*)calloc(len, sizeof(unsigned short));
    int i;
    for(i = 0; i < len; i++)
    {
      temp[(i + x) % len] = row[i];
    }
    return temp;
}


//Create a binary circular matrix
bin_matrix make_matrix_cpu(int rows, int cols, unsigned short* vec, int x)
{
    bin_matrix mat = mat_init_cpu(rows, cols);
    set_matrix_row_cpu(mat, 0, vec);
    int i;
    for(i = 1; i < rows; i++)
    {
      vec = shift_cpu(vec, x, cols);
      set_matrix_row_cpu(mat, i, vec);
    }
    return mat;
}

//Splice the row for the given range (does not include max)
ushort* splice_cpu(ushort* row, int min, int max)
{
    ushort* temp = (ushort*)calloc(max - min, sizeof(ushort));
    int i;
    for(i = min; i < max; i++)
    {
      temp[i - min] = row[i];
    }
    return temp;
}

//Constructing the pariy check matrix
bin_matrix parity_check_matrix_cpu(mdpc code)
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    bin_matrix H = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, 0, code->p), 1);
    int i;
    for(i = 1; i < code->n0; i++)
    {
      bin_matrix M = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, i * code->p, (i + 1) * code->p), 1);
      H = concat_horizontal_cpu(H, M);
    }
    end = clock();
    cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
    printf("Time for H: %f\n", cpu_time_used);
    // printf("H: \n");
    // print_matrix(H);
    //printf("Parity matrix generated...\n");
    return H;
}

//Constructing the generator matrix
bin_matrix generator_matrix_cpu(mdpc code)
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    bin_matrix H = parity_check_matrix_cpu(code);


    //End of modified code
    printf("Construction of G started...\n");
    bin_matrix H_inv = circ_matrix_inverse_cpu(make_matrix_cpu(code->p, code->p, splice_cpu(code->row, (code->n0 - 1) * code->p, code->n), 1));
    //printf("H_inv generated...\n");
    //printf("stop\n");
    bin_matrix H_0 = make_matrix_cpu(code->p, code->p, splice_cpu(code->row, 0, code->p), 1);
    bin_matrix Q = transpose_cpu(matrix_mult_cpu(H_inv,  H_0));
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

//Return the weight of the given row from the indices min to max
int get_row_weight(unsigned short* row, int min, int max)
{
    int weight = 0;
    int i;
    for(i = min; i < max + 1; i++) {
        if(row[i] == 1) {
            weight++;
        }
    }
    return weight;
}


mdpc qc_mdpc_init_cpu(int n0, int p, int w, int t)
{
    mdpc code;
    code = (mdpc)safe_malloc(sizeof(struct qc_mdpc));
    code->n0 = n0;
    code->p = p;
    code->w = w;
    code->t = t;
    code->n = n0 * p;
    code->r = p;
    code->k = (n0 - 1) * p;
    unsigned seed;
    code->row = (unsigned short*)calloc(n0 * p, sizeof(unsigned short));
    printf("Input seed or -1 to use default seed: ");
    scanf("%u", &seed);
    time_t tx;
    if(seed == -1) {
        srand((unsigned) time(&tx));
    } else {
        srand(seed);
    }

    while(1) {
        int flag = 0;
        int idx;
        while(flag < w)
        {
            idx = random_val(0, (n0 * p) - 1, seed);
            if(!code->row[idx])
            {
              code->row[idx] = 1;
              flag = flag + 1;
            }
        }
        if((get_row_weight(code->row, (n0 - 1) * p, (n0 * p)-1)) % 2 == 1)
        {
            break;
        }
        reset_row_cpu(code->row, 0, n0 * p);
    }
    printf("MDPC code generated....\n");
    return code;
}
