#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
using namespace std;

#define BLOCK_SIZE 16
#define BLOCK_SIZE2 256

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void fillA(double* A, int n, int k);
void fillb(double* b, double* A, int n, int k, int w);
void ReadFile(double* A, char* fileName, int n);
void solveOnDevice(double*A, double*b, int n, int k, int w);
void handleDeviceError(cudaError_t error);
void handleHostError(cudaError_t error);


__global__ void SetupMult(double* A, int n, int k, int half_k, int i){
  int tid = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  __shared__ double diag;

  if(threadIdx.y==0){
    diag = A[i*k + half_k];
  }
  __syncthreads();

  if(tid < half_k and i+1+tid < n)
    A[(i+1+tid)*k + half_k - 1 - tid] = A[(i+1+tid)*k + half_k - 1 - tid]/diag;
}

__global__ void LUDecomp(double* A, int n, int k, int half_k, int i){

  int tidx = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  int tidy = blockIdx.y*BLOCK_SIZE + threadIdx.y;

  int row = tidy + i + 1;
  int col = tidx + half_k - tidy;

  if(tidx<half_k && tidy<half_k && row<n && col < k){
    A[row*k + col] = A[row*k + col] - A[row*k + col - tidx - 1]*A[i*k + half_k + tidx + 1];
  }

}

__global__ void ForwardSolve(double* A, double* b, int n, int k, int half_k, int i){
  int ty = threadIdx.y;
  int by = blockIdx.y;
  int tidy = by*BLOCK_SIZE2+ty;
  int row = tidy + i + 1;
  __shared__ double mult;

  if(ty==0){
    mult = b[i];
  }

  __syncthreads();

  if(tidy < half_k && row < n){
    b[row] = b[row] - A[row*k + half_k - 1 - tidy]*mult;
  }

}

__global__ void BackSolve(double* A, double* b, int n, int k, int half_k, int i){
  int ty = threadIdx.y;
  int by = blockIdx.y;
  int tidy = by*BLOCK_SIZE2+ty;
  int row = i - 1 - tidy;
  __shared__ double mult;

  if(ty==0){
    b[i] = b[i]/A[i*k + half_k];
    mult = b[i];
  }

  __syncthreads();

  if(tidy < half_k && row >= 0){
    b[row] = b[row] - A[row*k + half_k + 1 + tidy]*mult;
  }

}







////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){

  int n;
  int k;
  int j;
  int w; //width of b

  switch(argc-1){
  case 3:
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    w = atoi(argv[3]);
    srand(119);
  break;
  case 5:
    n = atoi(argv[3]);
    k = atoi(argv[4]);
    w = atoi(argv[5]);
  break;
  default:
    printf("\nSupply exactly 3 or 5 arguments: See README_BUILD_AND_USAGE\n");
    exit(1);
  break;
  } 

  //ensure k is odd
  if(k%2 ==0){
    printf("\nBandwidth must be odd\n");
    exit(1);
  }

  //allocate matricies
  double *A, *b;
  cudaHostAlloc((void**)&A, n*k*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&b, n*w*sizeof(double), cudaHostAllocDefault);


 //fill matricies with either random or supplied values
  switch(argc-1){
    case 3:
      fillA(A, n, k);
      fillb(b, A, n, k, w);
    break;
    case 5:
      ReadFile(A, argv[1], n*k);
      ReadFile(b, argv[2], n*w);
    break;
  }

  for( int i = 0; i < n; i++) {
    for( int j = 0; j < n; j++) {
        printf("%.2lf ", A[i*n+j]);
    }
    printf("\n");
  }
  printf("\n");


  //solve the system of equations
  solveOnDevice(A,b,n,k,w);


  for( int i = 0; i < n; i++) {
    for( int j = 0; j < w; j++) {
        printf("%.2lf ", b[i*n+j]);
    }
    printf("\n");
  }
  printf("\n");



  //If running with random matricies, test results;
  //If running with supplied matricies, print output to stdout
  switch(argc-1){
    case 3:
      for(j=0; j<n*w; j++){ 
          if(fabs(b[j] - 1.0000) > 0.001f || b[j] != b[j]){
            printf("\nFailed at x[%i] = %f", j, b[j]);
            exit(1);
          }
      }
      printf("Test PASSED\n");
    break;
    case 5:
      printf("\nx\n");
        for(j=0; j<n*w; j++){
          if(j % w == 0) printf("\n");
          printf(" % 3.5f", b[j]);
        }
    break;
  }

   // Free matrices
  cudaFree(A);
  cudaFree(b);
  return 0;
}

void fillb(double* b, double* A, int n, int k, int w){
  int i, j, l;
  for(l=0; l<w; l++){
    for(i=0; i<n; i++){
      for(j=0; j<k; j++){
        b[i + n*l] += A[i*k + j];
      }
    }   
  }
}

//Created matrix garanteed to be diagonally dominent
void fillA(double* A, int n, int k){
   int half_k = (k-1)/2; //convert k from a bandwidth to a half bandwidth
   int i, j;
   for(i=0; i<n; i++){
     for(j=0; j<k; j++){
       if(j == half_k)
         A[i*k + j] = (double) rand()/RAND_MAX + (double) k;
       else if (j > half_k - 1 - i &&  j < half_k + n - i)
         A[i*k + j] = (double) rand()/RAND_MAX;
       else
         A[i*k + j] = 0.0;
     }
   }

}

void solveOnDevice(double *A, double *b, int n, int k, int w)
{
   int i, l;
   int half_k = (k-1)/2;
   float inclusiveTime;
   cudaEvent_t startInclusive, stopInclusive;
   cudaEventCreate(&startInclusive);
   cudaEventCreate(&stopInclusive);

   // Allocate device matricies
   double* Ad;
   double* bd;

   handleDeviceError(cudaMalloc((void**) &Ad, sizeof(double)*n*k));
   handleDeviceError(cudaMalloc((void**) &bd, sizeof(double)*n*w));

   cudaEventRecord(startInclusive, 0);//start inclusive timing 

   handleDeviceError(cudaMemcpyAsync(Ad, A, sizeof(double)*n*k, cudaMemcpyHostToDevice));


   //LU decomposition
   for(i=0; i<n-1; i++){
     dim3 dimBlock(1,BLOCK_SIZE,1);
     dim3 dimGrid(1,(half_k-1)/BLOCK_SIZE+1,1);
     SetupMult<<<dimGrid, dimBlock>>>(Ad, n, k, half_k, i);

     dim3 dimBlock2(BLOCK_SIZE,BLOCK_SIZE,1);
     dim3 dimGrid2((half_k -1)/BLOCK_SIZE+1, (half_k -1)/BLOCK_SIZE+1,1);
     LUDecomp<<<dimGrid2, dimBlock2 >>>(Ad, n, k, half_k, i);
   }

   dim3 dimBlock3(1,BLOCK_SIZE2,1);
   dim3 dimGrid3(1, (half_k-1)/BLOCK_SIZE2+1,1);

  for(l=0; l<w; l++){
     handleDeviceError(cudaMemcpyAsync(bd, b+l*n, sizeof(double)*n, cudaMemcpyHostToDevice));

     //Forward Solve
     for(i=0; i<n; i++){
       ForwardSolve<<<dimGrid3, dimBlock3>>>(Ad, bd, n, k, half_k, i);
     }
     //Back Solve
     for(i=n-1; i>=0; i--){
       BackSolve<<<dimGrid3, dimBlock3>>>(Ad, bd, n, k, half_k, i);
     }
     handleHostError(cudaMemcpyAsync(b + l*n, bd, sizeof(double)*n, cudaMemcpyDeviceToHost));
   }
 

   cudaEventRecord(stopInclusive, 0);  //stop inclusive timing
   cudaEventSynchronize(stopInclusive);//

   // Free device matrices
   cudaFree(Ad);
   cudaFree(bd);

   //calculate elapsed times
   cudaEventElapsedTime(&inclusiveTime, startInclusive, stopInclusive);

   //destroy timing events
   cudaEventDestroy(startInclusive);
   cudaEventDestroy(stopInclusive);
   printf("Inclusive GPU exection time (ms): %f\n", inclusiveTime);
}

void handleDeviceError(cudaError_t error){
  if(error != cudaSuccess){
    printf("Error allocating or copying memory to the device.");
    exit(1);
  }
}

void handleHostError(cudaError_t error){
  if(error != cudaSuccess){
    printf("Error allocating or copying memory to the host.");
    exit(1);
  }
}

void ReadFile(double* A, char* fileName, int n)
{
  printf("hello\n");
  int dataRead = n;
  std::ifstream ifile(fileName);

  for(unsigned int i = 0; i < n; i++){
    ifile>>A[i];
    dataRead--;
    printf("%i", i);
  }
  ifile.close();
}
