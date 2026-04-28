# Sgemv - Matrix times Vector

- sgemv_32: suitable for K∈[32,128]; for K smaller than 32 or larger than 128 there are further optimization methods.

Build and run:
```bash
nvcc sgemv_k32.cu -o sgemv_k32 -lcublas && sgemv_k32
```

# References
1. [GPU optimization series made simple: gemv optimization](https://zhuanlan.zhihu.com/p/494144694)
2. [GitHub - How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/sgemv/Sgemv_v0.cu)