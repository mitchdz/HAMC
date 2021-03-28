#include "../test.h"
#include "TransposeMatrix.cu"

bin_matrix solution(bin_matrix A){
    bin_matrix B;
    B = mat_init(A->cols, A->rows);
    for(int i = 0; i < A->rows; i++){
        for(int j = 0; j < A->cols; j++){
            set_matrix_element(B, j, i, mat_element(A, i, j));
        }
    }
    return B;    
}

void main(int argc, char *argv[]){
    int x, y, opt;
    while((opt = getopt(argc, argv, "x:y:")) != -1){
        switch(opt){
            case 'x':
                x = optarg;
                break;
            case 'y':
                y = optarg;
                break;
        }
    }
    
    bin_matrix A = mat_init(x, y);
    A->data = dataGen(A->rows, A->cols);
    bin_matrix B = mat_init(y, x);
    B->data = datGen(B->rows, B->cols);
    bin_matrix S = mat_init(x, y);
    S = solution(A);
    
    bool sol = true;
    for(int i = 0; i < A->rows * A->cols; i++){
        if(S->data[i] != B->data[i]){
            sol = false;
            cout << "Problem at: " << i << endl;
        }
    }
    cout << "Solution: " << sol << endl;
    free(A);
    free(B);
    free(S);
}