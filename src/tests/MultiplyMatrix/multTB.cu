#include "../test.h"
#include "MultiplyMatrix.cu"

bin_matrix solution(bin_matrix A, bin_matrix B){
    if (A->cols != B->rows){
        printf("Matrices are incompatible, check dimensions...\n");
        exit(0);
    }

    bin_matrix C;
    C = mat_init(A->rows, B->cols);
    bin_matrix B_temp = transpose(B);

    for(int i = 0; i < A->rows; i++){
        for(int j = 0; j < B->cols; j++){
            unsigned short val = 0;
            for(int k = 0; k < B->rows; k++){
                val = (val ^ (mat_element(A, i, k) & mat_element(B_temp, j, k)));
            }
            mat_element(C, i, j) = val;
        }
    }
    return C;
}

void main(int argc, char *argv[]){
    
    bin_matrix A = mat_init(x, y);
    A->data = dataGen(A->rows, A->cols);
    bin_matrix B = mat_init(y, x);
    B->data = datGen(B->rows, B->cols);
    bin_matrix S = mat_init(x, y);
    S = solution(A, B);
    bin_matrix C = mat_init(x, y);
    C = matrix_mult(A, B, C);
    //Check solution
    bool sol = true;
    for(int i = 0; i < A->rows * A->cols; i++){
        if(S->data[i] != C->data[i]){
            sol = false;
            cout << "Problem at: " << i << endl;
        }
    }
    cout << "Solution: " << sol << endl;
    free(A);
    free(B);
    free(C);
    free(S);
}