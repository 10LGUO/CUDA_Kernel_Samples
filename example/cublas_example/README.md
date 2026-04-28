# Matrix Multiplication with cuBLAS
> Reference blog post: https://www.cnblogs.com/cuancuancuanhao/p/7763256.html

When storing matrices, C/C++ is row-major while cuBLAS is column-major. Using the identity $AB={(B^TA^T)}^T$, adjust the inputs:
```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, *A, m, *B, k, &beta, *C, m);
                                              ↓  ↓              ↓  ↓   ↓                ↓
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, *B, n, *A, k, &beta, *C, n);
```

Build:
```bash
mkdir build
cd build
cmake ..
make
```

Run:
```bash
./cublas_exmple
```

Output:
```
C=
38.00   44.00   50.00   56.00
83.00   98.00   113.00  128.0
```