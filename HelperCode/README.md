# Helpful Hints about Cuda and Futhark Programming

## Lab 1: Simple CUDA Programming

A very simple helper CUDA code is provided in [Lab-1-Cuda](Lab-1-Cuda). In the first lab, your task is to extend the code to execute correctly on GPUs a program that multiplies each element of an array of arbitrary size with two.

Another thing that is good to hear early (and often) is that, in the context of GPUs' global memory, spatial locality means, under a slight simplification, that consecutive threads access consecutive memory locations. This is refer to as "coalesced access". More exactly: consecutive threads in a warp access a set of contiguous locations in global memory during a load instruction (executed in lockstep/SIMD). A funny exercise is to change the access pattern of the write so that consecutive threads access memory with a stride greater than 16. How much is the performance affected?

## Lab 2: List Homomorphisms in Futhark

A demonstration of how to integrate benchmarking and validation directly in Futhark programs is shown in [Lect-1-LH/mssp.fut](HelperCode/Lect-1-LH/mssp.fut). Go in the corresponding folder and try

```bash
$ futhark test --backend=cuda mssp.fut
```

Have the tests succeeded?

Next you can try to also benchmark, by:

```bash
$ futhark bench --backend=cuda mssp.fut
```

If runtimes are displayed then the program also validated (in the cases for which a reference result was specified). Now try reading the `mssp.fut`, in particular w.r.t. the multiple ways of specifying the input and reference datasets directly specified inside that file (in case you did not already look).   Understanding the automatic testing procedure will probably help you in benchmarking and validating the implementation of the Longest-Satisfying-Segment Problem (LSSP), which is the subject of the first weekly. 

Please also read the comment below function `mk_input` in `mssp.fut`. This dynamic casting of type sizes will be very useful later on, when we flatten parallelism in Futhark.

