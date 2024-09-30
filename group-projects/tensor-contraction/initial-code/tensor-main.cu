#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#include "tensor-kernels.cu.h"

#define GPU_RUNS    25

#define DIMSIZE 24
#define WORK    (2.0*DIMSIZE*DIMSIZE*DIMSIZE*DIMSIZE*DIMSIZE*DIMSIZE*DIMSIZE)
#define TILE    4

/////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}


void randomInit(float* data, int size) {
   for (int i = 0; i < size; ++i)
   data[i] = rand() / (float)RAND_MAX;
}

template<class T>
void tensorProd(T* A, T* B, T* C, const int len) {
  for(int a = 0; a < len; a++) {
    for(int b = 0; b < len; b++) {
      for(int c = 0; c < len; c++) {
        for(int i = 0; i < len; i++) {
          for(int j = 0; j < len; j++) {
            for(int k = 0; k < len; k++) {
                float acc = 0.0;
                for(int d=0; d<len; d++) {
                    acc += A4(A,len,a,i,j,d) * A4(B,len,b,c,k,d);
                }
                A6(C,len,a,b,c,i,j,k) = acc;
            }
          }
        }
      }
    }
  } 
}

template<class T>
bool validate(float* A,float* B, unsigned int sizeAB){
    for(int i = 0; i < sizeAB; i++)
      if (fabs(A[i] - B[i]) > 0.0005){
        printf("INVALID RESULT %d %f %f\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main() {
   // set seed for rand()
   srand(2006);
 
   // 1. allocate host memory for the two matrices
   unsigned int size_A = DIMSIZE * DIMSIZE * DIMSIZE * DIMSIZE;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = size_A;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);
 
   // 2. initialize host memory
   randomInit(h_A, size_A);
   randomInit(h_B, size_B);
    
   // 3. allocate device memory
   float* d_A;
   float* d_B;
   cudaMalloc((void**) &d_A, mem_size_A);
   cudaMalloc((void**) &d_B, mem_size_B);
 
   // 4. copy host memory to device
   cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
 
   // 5. allocate host memory for the result C
   unsigned int size_C = size_A * DIMSIZE * DIMSIZE;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C   = (float*) malloc(mem_size_C);
   float* seq_C = (float*) malloc(mem_size_C);
 
   // 6. allocate device memory for the result
   float *d_C;
   cudaMalloc((void**) &d_C, mem_size_C);
 
   // 7. compute sequential matrix multiplication
   {
      double elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      tensorProd<float>(h_A, h_B, seq_C, DIMSIZE);

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS); 
      printf("Sequential Naive Tensor Product runs in: %.2f microsecs\n", elapsed);
   }

   // execute the naive kernel
   {
      // setup execution parameters
      int  dim = (DIMSIZE + TILE - 1) / TILE;
      int  dimbl = dim * dim * dim; 
      dim3 block(TILE*TILE, TILE*TILE, 1);
      dim3 grid (dimbl, dimbl, 1);

      tensorProdNaiveKer<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, DIMSIZE);
      cudaDeviceSynchronize();

      cudaMemset(d_C, 0, mem_size_C);

      double elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      for(int q=0; q<GPU_RUNS; q++) {
        tensorProdNaiveKer<float, TILE> <<< grid, block >>>(d_A, d_B, d_C, DIMSIZE);
      }
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU Naive Tensor Product ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU Naive Tensor Product runs in: %.2f microsecs\n", elapsed);
      double gigaFlops = (WORK * 1.0e-3f) / elapsed; 
      printf( "GPU Naive Tensor Product Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n"
            , gigaFlops, elapsed, grid.x, grid.y); 
   }

   // execute the block+register tiled kernel for normalized layout
   {
      int  dim = (DIMSIZE + TILE - 1) / TILE;
      int  dimbl = dim * dim * dim; 
      dim3 block(TILE*TILE, TILE*TILE, 1);
      dim3 grid (dimbl, dimbl, 1);

      tensorProdTiledKerNorm<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, DIMSIZE);
      cudaDeviceSynchronize();

      cudaMemset(d_C, 0, mem_size_C);

      double elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      for(int q=0; q<GPU_RUNS; q++) {
        tensorProdTiledKerNorm<float, TILE> <<< grid, block >>>(d_A, d_B, d_C, DIMSIZE);
      }
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU Normalized-Layout Tensor Product ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU Normalized-Layout Tensor Product runs in: %.2f microsecs\n", elapsed);
      double gigaFlops = (WORK * 1.0e-3f) / elapsed; 
      printf( "GPU Normalized-Layout Tensor Product Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n"
            , gigaFlops, elapsed, grid.x, grid.y ); 
   }


   // 7. clean up memory
   free(h_A);
   free(h_B);
   free(h_C);
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
}

