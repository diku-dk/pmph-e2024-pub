= Second Weekly Assignment for the PMPH Course

This is the text of the second weekly assignment for the DIKU course
"Programming Massively Parallel Hardware", 2024-2025.

Hand in your solution in the form of a report in text or PDF
format, along with the missing code.   We hand-in incomplete code in 
the archive `w2-code-handin.tar.gz`.   You are supposed to fill in the missing
code such that all the tests are valid, and to report performance 
results. 

Please send back the same files under the same structure that was handed
in---implement the missing parts directly in the provided files.
There are comments in the source files that are supposed to guide you
(together with the text of this assignment).

Unziping the handed in archive `w2-code-handin.tar.gz` will create the
`w2-code-handin` folder, which contains two folders: `primes-futhark`
and `cuda`.

Folder `primes-futhark` contains the futhark source files related to Task 1,
prime number computation.

Folder `cuda` contains the cuda sorce files related to the other tasks:

* A `Makefile` that by default compiles and runs all programs, but the
    built-in validation may fail because some of the implementation is
    missing.

* Files `host_skel.cuh`, `pbb_kernels.cuh`, `pbb_main.cu` and `constants.cuh`
    contain the implementation of reduce and (segmented) scan inclusive. The
    implementation is generic---one may change the associative binary
    operator and type; take a look---and allows one to fuse the result
    of a `map` operation with the reduce/scan.

* Files `spmv_mul_kernels.cuh` and `spmv_mul_main.cu` contain the
    incomplete implementation of the sparse-matrix vector multiplication.

When implementing the CUDA tasks, please take a good look around you to
see where and how those functions are used in the implementation of 
`reduce` and/or `scan`.

*Important Observation for the CUDA tasks:*

1. The value of `ELEMS_PER_THREAD` (which must be defined as a statically-known
   constant in file `constants.cuh`)  needs to be adjusted to the
   value of the CUDA block size `B`:

    * For `B = 256` one can use `ELEMS_PER_THREAD = 24` (best performance)

    * For `B = 512` one can use `ELEMS_PER_THREAD = 12`

    * For `B =1024` one can use `ELEMS_PER_THREAD = 6` (I think `8` works as well)

    * Using `B = 1024` and `ELEMS_PER_THREAD = 12` will result in an error caused by
      certain kernels running out of shared memory.
 
2. Best performance requires increasing the value of `RUNS_GPU` in file `constants.cuh` 
   to `500` or `1000`. But do this only as the final step when you measure the
   performance, not during development: there are many of you and only 3 GPUs.

3. The `hendrixfut03fl` server has two GPUs; you can run on the second one by
   adding `cudaSetDevice(1);` at the begining of the `main` function(s). 
    
== Task 1: Flat Implementation of Prime-Numbers Computation in Futhark (3 pts)

This task refers to flattening the nested-parallel version of prime-number 
computation, which computes all the prime numbers less than or equal to `n`
with `D(n) = O(lg lg n)`.   More detail can be found in lecture slides `L2-Flatenning.pdf`
and lecture notes, section 4.3.2 entitled 
"Exercise: Flattening Prime Number Computation (Sieve)".

Please also read section 3.2.5 from lecture notes entitled 
"Prime Number Computation (Sieve)" which describes two versions: one flat
parallel one, but which has sub-optimal depth, and the nested-parallel one
that you are supposed to flatten.  These versions are fully implemented in
files `primes-naive.fut` and `primes-seq.fut`, respectively. The latter
corresponds morally to the nested-parallel version, except that it has 
been implemented with (sequential) loops, because  Futhark does not 
support irregular parallelism.  

A third fully-implemented code version, named `primes-adhoc.fut`, is provided:
this has optimal depth and very regular/efficient `map` parallelism, but has
suboptimal work complexity.
The third verson is meant to demonstrate that one cannot fight complexity
with parallelism, i.e., when you write a parallel implementation make sure
that it preserves the optimal sequential work asymptotics, a.k.a., is *work efficient*.

*Your task is to:*

* fill in the missing part of the implementation in file `primes-flat.fut`;

* make sure that `futhark test --backend=cuda primes-flat.fut`
    succeeds on the small dataset;
    
* a large dataset `N=10000000` is declared for automatic benchmarking
  in all Futhark implementation of prime computation. Please compile and
  run one of the already-implementated versions (e.g., `primes-seq.fut`)
  to generate a reference result named `ref10000000.out` for this large
  dataset and make it part of the automatic validation of your
  implementation (e.g., remove a line that breaks the contiguous
  comments used for validation).
  To generate said reference result, compile and run with:
  
  ```
  $ futhark c primes-seq.fut
  $ echo "10000000i64" | ./primes-seq -b > ref10000000.out
  ```

* measure the runtimes corresponding to the large datset for all four
  implementations,
  i.e., the three already provided and the one you just implemented.
  Please use `futhark bench --backend=c` for `primes-seq.fut`, since
  this does not uses any parallelism, and `futhark bench --backend=cuda`
  for the other three.

*In your report:*

* please state whether your implementation validates on both datasets,

* please present the code that you have added and briefly explain it,
  i.e., its correspondence to the flattening rules,
  
* please report the runtimes of all four code versions for the large
  dataset and try to briefly explain them, i.e.,
  Do they match what you would expect from their work-depth complexity?


== Task 2: Copying from/to Global to/from Shared Memory in Coalesced Fashion (2pt)

This task refers to improving the spatial locality of global-memory accesses.

On NVIDIA GPUs, threads are divided into warps, in which a warp contains
`32` consecutive threads. The threads in the same warp execute in lockstep
the same SIMD instruction. 

"Coalesced" access to global memory means that the threads in a warp
will access (read/write) in the same SIMD instruction consecutive
global-memory locations. This coalesced access is optimized by hardware
and requires only one (or two) memory transaction(s) to complete
the corresponding load/store instruction for all threads in the same warp.
Otherwise, as many as `32` different memory transactions may be executed
sequentially, which results in a significant overhead.

Your task is to modify in the implementation of `copyFromGlb2ShrMem` and
`copyFromShr2GlbMem` functions in file `pbb_kernels.cuh` the (one) line that
computes `loc_ind`---i.e.,`uint32_t loc_ind = threadIdx.x * CHUNK + i;`---such
that the new implementation is semantically equivalent to the existent one, 
but it features coalesced access (read/write) to global memory. 
(The provided implementation exhibits uncoalesced access; read more in the
comments associated with function `copyFromGlb2ShrMem`.)

*Please write in your report:*

* your one-line replacement;

* briefly explain why your replacement ensures coalesced access to global memory;

* explain to what extent your one-line replacement has affected the performance,
    i.e., which tests and by what factor.

I suggest to do the comparison after you also implement Task 3.
You may keep both new and old implementations around for both Tasks 2 and 3
and select between them statically by `#if 1 #then ... #else ... #endif`.
Then you can reason separately what is the impact of Task 2 optimization
and of Task 3 optimization.

== Task 3: Implement Inclusive Scan at WARP Level (2 pts)

This task refers to implementing an efficient WARP-level scan in function
`scanIncWarp` of file `pbb_kernels.cuh`, i.e., each warp of threads scans,
independently of other warps, its `32` consecutive elements stored in 
shared memory.  The provided (dummy) implementation works correctly, 
but it is very slow because the warp reduction is performed sequentially 
by the first thread of each warp, so it takes `WARP-1 == 31` steps to 
complete, while the other `31` threads of the WARP are idle.

Your task is to re-write the warp-level scan implementation in which
the threads in the same WARP cooperate such that the depth of
your implementation is 5 steps ( WARP==32, and lg(32)=5 ).
The algorithm that you need to implement, together with
some instructions is shown in document `Lab2-RedScan.pdf`---the 
slide just before the last one. 
The implementation does not need any synchronization, i.e.,
please do NOT use `__syncthreads();` and the like in there
(it would break the whole thing!).

*Please write in your report:*

* the full code of your implementation of `scanIncWarp`
    (should not be longer than 20 lines)

* explain the performance impact of your implementation:
    which tests were affected and by what factor. Does the
    impact becomes higher for smaller array lengths?

(I suggest you correctly solve Task 2 before measuring the impact.
The optimizations of Task 2 and 3 do not apply for all tests, so some
will not benefit from it.)

== Task 4: Find the bug in `scanIncBlock`  (1 pts)

There is a nasty race-condition bug in function `scanIncBlock` of file `pbb_kernels.cuh`
which appears only for CUDA blocks of size 1024. For example running from terminal with
command `./test-pbb 100000 1024` should manifest it. 
(Or set 1024 as the second argument of `test-pbb` in `Makefile`.)

Can you find the bug? This will help you to understand 
how to scan CUDA-block elements with a CUDA-block of threads by piggy-backing
on the implementation of the warp-level scan that is the subject of Task 3. 
It will also shed insight in GPU synchronization issues.  

*Please explain in the report the nature of the bug, why does it appear only
    for block size 1024, and how did you fix it.*

*When compiling/running with block size `1024` remember* to set the value of 
`ELEMS_PER_THREAD` (in file `constants.cuh`) to `6` otherwise you will also get
other errors!

== Task 5: Flat Sparse-Matrix Vector Multiplication in CUDA (2 pts)

This task refers to writing a flat-parallel version of sparse-matrix vector multiplication in CUDA.
Take a look at Section 3.2.4 ``Sparse-Matrix Vector Multiplication'' in lecture notes, page 40-41 
and at section 4.3.1 ``Exercise: Flattening Sparse-Matrix Vector Multiplication''.

*Your task is to:*

* implement the four kernels of file  `spmv_mul_kernels.cuh` and two lines in file `spmv_mul_main.cu` (at lines 155-156).

* run the program and make sure it validates.

* add your implementation in the report (it is short enough) and report speedup/slowdown vs sequential CPU execution.
