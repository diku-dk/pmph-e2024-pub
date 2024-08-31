#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 100

__global__ void mul2Kernel(float* X, float *Y) {
    const unsigned int gid = threadIdx.x;
    Y[gid] = 2 * X[gid];
}

int main(int argc, char** argv) {
    unsigned int N;
    
    { // reading the number of elements 
      if (argc != 2) { 
        printf("Num Args is: %d instead of 1. Exiting!\n", argc); 
        exit(1);
      }

      N = atoi(argv[1]);
      printf("N is: %d\n", N);

      const unsigned int maxN = 500000000;
      if(N > maxN) {
          printf("N is too big; maximal value is %d. Exiting!\n", maxN);
          exit(2);
      }
    }

    // use the first CUDA device:
    cudaSetDevice(0);

    unsigned int mem_size = N*sizeof(float);

    // allocate host memory
    float* h_in  = (float*) malloc(mem_size);
    float* h_out = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=0; i<N; ++i) {
        h_in[i] = (float)i;
    }

    // allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        mul2Kernel<<< 1, N>>>(d_in, d_out);
    }
  
    { // execute the kernel a number of times;
      // to measure performance use a large N, e.g., 200000000,
      // and increase GPU_RUNS to 100 or more. 
    
        double elapsed; struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int r = 0; r < GPU_RUNS; r++) {
            mul2Kernel<<< 1, N>>>(d_in, d_out);
        }
        cudaDeviceSynchronize();
        // ^ `cudaDeviceSynchronize` is needed for runtime
        //     measurements, since CUDA kernels are executed
        //     asynchronously, i.e., the CPU does not wait
        //     for the kernel to finish.
        //   However, `cudaDeviceSynchronize` is expensive
        //     so we need to amortize it across many runs;
        //     hence, when measuring performance use a big
        //     N and increase GPU_RUNS to 100 or more.
        //   Sure, it would be better by using CUDA events, but
        //     the current procedure is simple & works well enough.
        //   Please note that the execution of multiple
        //     kernels in Cuda executes correctly without such
        //     explicit synchronization; we need this only for
        //     runtime measurement.
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / GPU_RUNS;
        double gigabytespersec = (2.0 * N * 4.0) / (elapsed * 1000.0);
        printf("The kernel took on average %f microseconds. GB/sec: %f \n", elapsed, gigabytespersec);
        
    }
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    // print result
    //for(unsigned int i=0; i<N; ++i) printf("%.6f\n", h_out[i]);

    for(unsigned int i=0; i<N; ++i) {
        float actual   = h_out[i];
        float expected = 2 * h_in[i]; 
        if( actual != expected ) {
            printf("Invalid result at index %d, actual: %f, expected: %f. \n", i, actual, expected);
            exit(3);
        }
    }
    printf("Successful Validation.\n");

    // clean-up memory
    free(h_in);       free(h_out);
    cudaFree(d_in);   cudaFree(d_out);
}
