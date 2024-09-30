#ifndef TENSOR_KERNELS
#define TENSOR_KERNELS

#define A4(X,len,I1,I2,I3,I4) (X[ (I1)*(len)*(len)*(len) + \
                                  (I2)*(len)*(len) + \
                                  (I3)*(len) + (I4) ])

#define A6(X,len,I1,I2,I3,I4,I5,I6) (X[ (I1)*(len)*(len)*(len)*(len)*(len) + \
                                        (I2)*(len)*(len)*(len)*(len) + \
                                        (I3)*(len)*(len)*(len) + \
                                        (I4)*(len)*(len) + (I5)*(len) + (I6) ])

template <class ElTp, int T> 
__global__ void tensorProdNaiveKer(ElTp* A, ElTp* B, ElTp* C, const int len) {
  int i, ii, j, jj, aa, k, kk, c, cc, bb;
  int tmp;

  { // compute i, j, a, k, c, b
    c = threadIdx.y / T;
    i = threadIdx.y % T;

    j = threadIdx.x / T;
    k = threadIdx.x % T;
  }

  { // compute ii, jj, aa, kk, cc, bb
    int num_d    = (len + T - 1) / T;
    int num_d_sq = num_d*num_d;

    aa = (blockIdx.y / num_d_sq) * T;
    tmp = blockIdx.y % num_d_sq;
    bb = (tmp / num_d) * T;
    cc = (tmp % num_d) * T;

    ii = (blockIdx.x / num_d_sq) * T;
    tmp = blockIdx.x % num_d_sq;
    jj = (tmp / num_d) * T;
    kk = (tmp % num_d) * T;
  }

  if ( (j+jj >= len) || (i+ii >= len) ||
       (k+kk >= len) || (c+cc >= len)  )
    return; // out of range

  for(int b=0; b<T; b++) {
    for(int a=0; a<T; a++) {
      ElTp accum = 0.0;
      for(int d=0; d<len; d++) {
        ElTp x = (aa+a<len)? A4(A,len,aa+a,ii+i,jj+j,d) : 0.0;
        ElTp y = (bb+b<len)? A4(B,len,bb+b,cc+c,kk+k,d) : 0.0;
        accum +=  x * y;
      }
      if( (aa+a<len) && (bb+b <len) )
        A6(C,len,aa+a,bb+b,cc+c,ii+i,jj+j,kk+k) = accum;
    }
  }
}


template <class ElTp, int T> 
__global__ void tensorProdTiledKerNorm(ElTp* A, ElTp* B, ElTp* C, const int len) {
    __shared__ ElTp Aloc[T][T][T][2*T];
    __shared__ ElTp Bloc[T][T][T][2*T];
    int i, ii, j, jj, aa, k, kk, c, cc, bb;
    int tmp;

    { // compute i, j, a, k, c, b
        c = threadIdx.y / T;
        i = threadIdx.y % T;

        j = threadIdx.x / T;
        k = threadIdx.x % T;
    }

    { // compute ii, jj, aa, kk, cc, bb
        int num_d    = (len + T - 1) / T;
        int num_d_sq = num_d*num_d;

        aa = (blockIdx.y / num_d_sq) * T;
        tmp = blockIdx.y % num_d_sq;
        bb = (tmp / num_d) * T;
        cc = (tmp % num_d) * T;

        ii = (blockIdx.x / num_d_sq) * T;
        tmp = blockIdx.x % num_d_sq;
        jj = (tmp / num_d) * T;
        kk = (tmp % num_d) * T;
    }

    ElTp Areg[T];
    ElTp Breg[T];
    ElTp Creg[T][T];

    #pragma unroll
    for(int a=0; a<T; a++) {
        #pragma unroll
        for(int b=0; b<T; b++) {
            Creg[b][a] = 0.0;
        }
    }

    for(int dd=0; dd<len; dd+=2*T) {
        { // copy slice of A from global to local memory (coalesced on d)
            #pragma unroll
            for(int q=0; q<2; q++) {
                ElTp elm = 0.0;
                bool safeA = (aa+c < len) || (ii+i < len) || (jj+j/2+q*(T/2) < len) || (dd+(j%2)*T+k < len);
                if( safeA ) {
                    //elm = A4(A, len, aa+c, ii+i, jj + j, dd + k); // need of LMAD copy
                    elm = A4(A,len,aa+c,ii+i,jj + j/2 + q*(T/2), dd + (j%2)*T + k); 
                }
                Aloc[c][i][j/2 + q*(T/2)][(j%2)*T + k] = elm;
            }
        }
        { // copy slice of B from global to local memory (coalesced on d)
            #pragma unroll
            for(int q=0; q<2; q++) {
                ElTp elm = 0.0;
                bool safeB = (bb+c < len) || (cc+i < len) || (kk + j/2 + q*(T/2) < len) || (dd + (j%2)*T + k < len);
                if( safeB ) {
                    //elm = A4(B, len, bb+c, cc+i, kk+j, dd+k); // need of LMAD copy
                    elm = A4(B,len,bb+c, cc+i, kk + j/2 + q*(T/2), dd + (j%2)*T + k);
                }
                Bloc[c][i][j/2 + q*(T/2)][(j%2)*T + k] = elm;
            }
        }
        __syncthreads();

        for(int d=0; d<2*T; d++) {
            // copy slice of A from local to register memory
            #pragma unroll
            for(int a=0; a<T; a++) {
                Areg[a] = Aloc[a][i][j][d];
            }
            // copy slice of B from local to register memory
            #pragma unroll
            for(int b=0; b<T; b++) {
                Breg[b] = Bloc[b][c][k][d];
            }

            #pragma unroll
            for(int a=0; a<T; a++) {
                #pragma unroll
                for(int b=0; b<T; b++) {
                    Creg[a][b] += Areg[a] * Breg[b];
                }
            }

        }
        __syncthreads();
    }

    #pragma unroll
    for(int a=0; a<T; a++) {
        #pragma unroll
        for(int b=0; b<T; b++) {
            A6(C,len,aa+a,bb+b,cc+c,ii+i,jj+j,kk+k) = Creg[a][b];
        }
    }
}

#endif // TENSOR_KERNELS
