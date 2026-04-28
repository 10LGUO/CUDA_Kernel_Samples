# CUDA_Kernel_Samples

## Introduction

This project is a hands-on guide to implementing CUDA operators:

1. It collects common CUDA operator problems and optimization strategies, with example implementations for each operator.
2. Every operator includes complete code from a naive implementation to optimized versions, making it easy to debug and analyze performance.
3. Each operator comes with the relevant GPU background knowledge to help you learn CUDA programming efficiently.

It currently covers the following common CUDA operators and their optimized versions:


| Folder      | Description                | Contents                          | Importance |
| ----------- | -------------------------- | --------------------------------- | ---------- |
| example     | Some simple examples       | /                                 | /          |
| elementwise | Element-wise array compute | add                               | Low        |
| gemv        | Matrix-vector multiply     | sgemv                             | Low        |
| reduce      | Reduction optimizations    | sum, max, softmax, softmax_matrix | High       |
| sgemm       | Matrix multiply opt.       | naive, blocktile, threadtile, ... | Medium     |
| transpose   | Matrix transpose opt.      | naive, coalesced access + bank conflict fix | Medium |


## Notes on Implementing Operators

Usually you only need to write the CUDA kernel function itself (in most cases this is all that matters), along with `block_size`, `grid_size`, and the kernel launch.

Here are some macros used later in this document:

```cpp
// 1. Ceiling division
#define CEIL(a, b) ((a + b - 1) / (b))

// 2. FLOAT4, used for vectorized memory access; either form works
// C style
#define FLOAT4(value) *(float4*)(&(value))

// C++ style
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
```

**The rest of this document takes this perspective and shows the essential code for reference and practice.**

# elementwise

**Importance**: **Low**

**Operator description**: elementwise is the simplest **class of operators**. It refers to operating on data element by element, for example adding the corresponding elements of two equal-length arrays ([add](./elementwise/add.cu)). Also, in deep learning an activation function computes an activation value for each element of its input, so activation functions fall within the elementwise category too.

There are mainly two ways to write these operators:

1. naive: each thread handles the computation of a single element.
2. Using **float4** and other vectorized memory-access methods: this only speeds things up for large-scale data. Note that you must **divide by 4 on the grid**, not on the block, otherwise you may reduce SM occupancy (Occupancy = actually running threads / theoretical max threads per SM). See 👉 [choosing grid_size and block_size](https://blog.csdn.net/LostUnravel/article/details/135721041) (block_size should be no smaller than the max threads per SM divided by the max concurrent blocks, and should be a divisor of the max threads per SM, to have a chance of reaching 100% Occupancy). The benefit of vectorized access is that it reduces the number of memory-access instructions, reads more data per unit time, and increases memory bandwidth.

**Source folder**: [./elementwise](./elementwise)

## add

Source: [./elementwise/add.cu](./elementwise/add.cu)

### naive version

```cpp
// block_size, grid_size, and the kernel launch
int block_size = 1024;
int grid_size  = CEIL(N, block_size);
elementwise_add<<<grid_size, block_size>>>(a, b, c, N);

// kernel definition
__global__ void elementwise_add(float* a, float* b, float *c, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Using vectorized memory access

To optimize with vectorized memory access, note that you must **divide by 4 on the grid**:

```cpp
int block_size = 1024;
int grid_size  = CEIL(CEIL(N,4), block_size);  // Note: divide by 4 on the grid dimension
elementwise_add<<<grid_size, block_size>>>(a, b, c, N);

__global__ void elementwise_add_float4(float* a, float* b, float *c, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    if (idx < N) {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b = FLOAT4(b[idx]);
        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;
        FLOAT4(c[idx]) = tmp_c;
    }
}
```

The block_size, grid_size, and kernel launch for the following operators are written the same way as for add, so they are not repeated.

## sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}} $$

```cpp
__global__ void sigmoid(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// float4
__global__ void sigmoid_float4(float* x, float* y, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = 1.0f / (1.0f + expf(-tmp_x.x));
        tmp_y.y = 1.0f / (1.0f + expf(-tmp_x.y));
        tmp_y.z = 1.0f / (1.0f + expf(-tmp_x.z));
        tmp_y.w = 1.0f / (1.0f + expf(-tmp_x.w));
        FLOAT4(y[idx]) = tmp_y;
    }
}
```

## relu

$$ \text{ReLU}(x) = \max(0, x) $$

```cpp
__global__ void relu(float* x, float* y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
    }

// float4
__global__ void relu_float4(float* x, float* y, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = fmaxf(0.0f, tmp_x.x);
        tmp_y.y = fmaxf(0.0f, tmp_x.y);
        tmp_y.z = fmaxf(0.0f, tmp_x.z);
        tmp_y.w = fmaxf(0.0f, tmp_x.w);
        FLOAT4(y[idx]) = tmp_y;
    }
}
```

# reduce

**Importance**: **High**

**Operator description**: reduce is an aggregation operation, typically used to reduce a multi-element data structure (such as an array or tensor) into a smaller one (usually a single value or a smaller array) according to some rule. It is widely used in data processing, parallel computing, and deep learning. Examples include summing an array (sum), taking the mean (mean), taking the maximum (max), and computing softmax. Among these, **sum and softmax are the most important**.

**Source folder**: [./reduce](./reduce)

## sum

Source: [./reduce/sum/sum.cu](./reduce/sum/sum.cu)

### naive version

Each thread writes to the same global memory location via the atomic function `atomicAdd`. Atomic functions serialize threads, losing parallelism and greatly reducing operator performance, so they should not be overused:

```cpp
dim3 block_size(BLOCK_SIZE);  // BLOCK_SIZE is some number defined by a macro
dim3 grid_size(CEIL(N, BLOCK_SIZE));
reduce_v1<<<grid_size, block_size>>>(d_x, d_y, N);

__global__ void reduce_v1(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) atomicAdd(output, input[idx]);
}
```

### Halving reduction

Perform a halving reduction within a block. Each block reduces one portion: first move it into the block's own shared memory, then reduce down to the first element.

> The drawback of this method is that BLOCK_SIZE must be a power of 2, otherwise the halving operation computes incorrectly and produces large errors. Also, each halving iteration must use `__syncthreads()` to synchronize, forcing all threads to wait at the sync point until the other threads in the block also arrive, which hurts performance.

```cpp
dim3 block_size(BLOCK_SIZE);  // BLOCK_SIZE is some number defined by a macro
dim3 grid_size(CEIL(N, BLOCK_SIZE));
reduce_v2<<<grid_size, block_size>>>(d_x, d_y, N);

__global__ void reduce_v2(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float input_s[BLOCK_SIZE];

    // 1. Move a number of elements equal to the thread count (blockDim.x) into this block's shared memory
    input_s[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // 2. Use 1/2, 1/4, 1/8... of the threads to perform the halving reduction
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {  // 2. halving reduction
            input_s[tid] += input_s[tid + offset];
        }
        __syncthreads();
    }

    // 3. The first thread of each block accumulates the result into the output
    if (tid == 0) atomicAdd(output, input_s[0]);
}
```

### warp shuffle (recommended)

Perform the halving reduction within a warp. The advantage is that threads within a warp are synchronized, so compared to halving at the block level, halving at the warp level does not need `__syncthreads()` on each step, giving higher parallelism.

> BLOCK_SIZE must be a multiple of 32, otherwise a warp with fewer than 32 threads is produced, which may access invalid data.

**Using the warp shuffle operations provided by CUDA**, the following functions are available:

1. `__shfl_sync()`: copy the value from any laneId (0~31) thread.
2. `__shfl_xor_sync()`: copy the value from a computed laneId (0~31) thread.
3. `__shfl_up_sync()`: copy the value from a thread with a smaller laneId by a given offset.
4. `__shfl_down_sync()`: copy the value from a thread with a larger laneId by a given offset.

Of these, `__shfl_xor_sync()` and `__shfl_down_sync()` are the most frequently used.

```cpp
dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CEIL(N, BLOCK_SIZE));
reduce_v3<<<grid_size, block_size>>>(d_x, d_y, N)

__global__ void reduce_v3(float* d_x, float* d_y, const int N) {
    __shared__ float s_y[32];  // only 32 needed, because a block has at most 1024 threads, i.e. at most 1024/32=32 warps

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;  // which warp this thread belongs to
    int laneId = threadIdx.x % warpSize;  // this thread's index within the warp

    float val = (idx < N) ? d_x[idx] : 0.0f;  // move d_x[idx] into this thread's register
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);   // halving reduction within a warp
    }

    if (laneId == 0) s_y[warpId] = val;  // the first thread of each warp stores the data into shared mem
    __syncthreads();

    if (warpId == 0) {  // use the first warp of each block for the final reduction of s_y
        int warpNum = blockDim.x / warpSize;  // number of warps in each block
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (laneId == 0) atomicAdd(d_y, val);  // the first thread of this warp accumulates the result into the output
    }
}
```

### warp shuffle + float4

A further optimization on warp shuffle: use float4 when moving data:

```cpp
#define FLOAT4(value) (float4*)(&(value))[0]
dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CEIL(CEIL(N, BLOCK_SIZE),4));  // divide by 4 here
reduce_v3<<<grid_size, block_size>>>(d_x, d_y, N)

__global__ void reduce_v4(float* d_x, float* d_y, const int N) {
    __shared__ float s_y[32];
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;  // multiply by 4 here
    int warpId = threadIdx.x / warpSize;   // which warp this thread is in
    int laneId = threadIdx.x % warpSize;   // this thread's index within the warp
    float val = 0.0f;
    if (idx < N) {
        float4 tmp_x = FLOAT4(d_x[idx]);
        val += tmp_x.x;
        val += tmp_x.y;
        val += tmp_x.z;
        val += tmp_x.w;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (laneId == 0) s_y[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (landId == 0) atomicAdd(d_y, val);
    }
}
```

## SoftMax

Both the CPU and CUDA implementations of Softmax are important to know.

Source: [./reduce/softmax/softmax.cu](./reduce/softmax/softmax.cu)

The Softmax formula is as follows:

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

To avoid overflow, we usually subtract the maximum value, so the following formula is commonly used:

$$
\text{Softmax}(x_i) = \frac{e^{x_i-M}}{\sum_{j=1}^{N} (e^{x_j-M})}
$$

where $M$ is the maximum value of the input vector.

### CPU version

```cpp
void softmax(float* input, float* output, int N) {
    int M = *(std::max_element(input, input + N));
    float div = 0;
    for (int i = 0; i < N; i++) {
        output[i] = std::exp(input[i] - M);
        div += output[i];
    }
    for (int i = 0; i < N; i++) {
        output[i] /= div;
    }
}
```

### CUDA version

The most direct idea is to split the Softmax computation into several reduction operators. As long as you can write a reduction, you can write Softmax.

The advantage of this approach is that it is fairly simple. Although there is more code, it is essentially reduction code, and the logic of the several operators does not differ much. The drawback is that the operator efficiency is relatively low. **Here it is recommended to study the [softmax_matrix](#softmax_matxrix) implementation!**

Idea:

- Kernel 1: reduce to find the maximum value max_val
- Kernel 2: reduce to compute the sum
- Kernel 3: for each element, subtract max_val and divide by sum.

```cpp
__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(float* input, float* max_val, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    float val = (idx < N) ? input[idx] : (-FLT_MAX);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    if (laneId == 0) s_mem[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : (-FLT_MAX);
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (laneId == 0) atomicMax(max_val, val);
    }
}

__global__ void sum_kernel(float* input, float* sum, float* max_val, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    float val = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (laneId == 0) s_mem[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (laneId == 0) atomicAdd(sum, val);
    }
}

__global__ void softmax_kernel(float* input, float* output, float* sum, float* max_val, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) output[idx] = expf(input[idx] - *max_val) / (*sum);
}

// Initialize the relevant variables
// ...
// Launch
int block_size = 256;
int grid_size  = CEIL(N, block_size);
max_kernel<<<gird_size, block_size>>>(input, max_val, N);
sum_kernel<<<gird_size, block_size>>>(input, sum, max_val, N);
softmax_kernel<<<gird_size, block_size>>>(input, output, sum, max_val, N);
```

# transpose

**Importance**: **Medium**

**Operator description**: this refers to matrix transpose, which involves efficient access to GPU global memory and the bank-conflict concept.

How to optimize global memory access:

1. **Coalesce accesses as much as possible**, i.e. consecutive threads read consecutive memory, and try to make the base address of the accessed global memory a multiple of 32 bytes (the amount of data handled by one data transfer) (cudaMalloc allocates at least a multiple of 256 bytes).
2. If you cannot coalesce both reads and writes simultaneously, you should **try to coalesce the writes**, because if the compiler can determine that a global memory variable is read-only inside the kernel, it will automatically use `__ldg()` to read global memory and cache the data, mitigating the impact of non-coalesced access. But this only works for reads; there is no equivalent for writes. Also, for the Kepler and Maxwell architectures you need to explicitly use `__ldg()`, e.g. `B[ny * N + nx] = __ldg(&A[nx * N + ny])`.

**Source folder**: [./transpose](./transpose)

## naive

```cpp
__global__ void transpose(float* input, float* output, int M, int N) {
    // row and col of input
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}
```

## Coalesced writes only

```cpp
__global__ void transpose(float* input, float* output, int M, int N) {
    // row and col of output
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < M) {
        output[row * M + col] = __ldg(&input[col * N + row]);  // coalesced writes, reads cached via __ldg
    }
}
```

## (Recommended) Use shared memory as a staging area, coalescing both reads and writes

shareMem

Note that this approach hits the classic bank-conflict problem when reading shared-memory data, which can be solved by padding or swizzling:

Padding the shared memory:

```cpp
// The input matrix is M rows by N cols; the output matrix is N rows by M cols
dim3 block(32, 32);
dim3 grid(CEIL(N,32), CEIL(M,32));  // tile according to input's shape (M rows, N cols)
transpose<32><<<grid, block>>>(input, output, M, N);

template <const int BLOCK_SIZE>
__global__ void transpose(float* input, float* output, int M, int N) {
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE + 1];  // padding
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int x1 = bx + threadIdx.x;
    int y1 = by + threadIdx.y;

    if (x1 < N && y1 < M) {
        s_mem[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
    }
    __syncthreads();

    int x2 = by + threadIdx.x;
    int y2 = bx + threadIdx.y;
    if (x2 < M && y2 < N) {
        output[y2 * M + x2] = s_mem[threadIdx.x][threadIdx.y];  // after padding, there is no bank conflict here
    }
}
```

Using swizzling, no padding of shared memory is needed:

```cpp
// The input matrix is M rows by N cols; the output matrix is N rows by M cols
dim3 block(32, 32);
dim3 grid(CEIL(N,32), CEIL(M,32));  // tile according to input's shape (M rows, N cols)
transpose<32><<<grid, block>>>(input, output, M, N);

template <const int BLOCK_SIZE>
__global__ void transpose(float* input, float* output, int M, int N) {
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE];  // no padding needed
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int x1 = bx + threadIdx.x;
    int y1 = by + threadIdx.y;

    if (x1 < N && y1 < M) {
        s_mem[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y1 * N + x1];
    }
    __syncthreads();

    int x2 = by + threadIdx.x;
    int y2 = bx + threadIdx.y;
    if (x2 < M && y2 < N) {
        output[y2 * M + x2] = s_mem[threadIdx.x][threadIdx.x ^ threadIdx.y];  // after swizzling, there is no bank conflict here
    }
}
```

# sgemm

**Importance**: **Medium**

**Operator description**: this refers to matrix multiplication. Matrix multiply is a classic example when learning CUDA and involves many optimization techniques commonly used in CUDA programming. It is recommended to read [./sgemm/README.md](./sgemm/README.md). Writing it from scratch is often quite difficult, so it is recommended to first master the simplest naive version and the block_tile version. Once you have mastered the block_tile version, you only need to add a bit of code to optimize it into the thread_tile version, so it is also worth mastering. For the remaining, more efficient optimized versions, understanding their principles is generally sufficient.

**Source folder**: [./sgemm](./sgemm)

## naive version

```cpp
// C(MxN) = A(MxK) * B(KxN), row-major
// Each thread handles one element of the output matrix

// Assume M N K have been assigned
const int BLOCK_SIZE = 32;
dim3 block(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));  // tile according to C's shape (M rows, N cols)
sgemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float accum = 0.0f;
    for (int i = 0; i < K; i++) {
        accum += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = accum;
}
```

## block_tile version

Each thread still computes one element of the output matrix, but shared memory is used as a cache to read repeatedly from shared mem instead of global mem. The number of reads does not decrease, but shared mem is faster to read than global mem:

```cpp
#define BLOCK_SIZE 32

dim3 block(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(CEIL(N,BLOCK_SIZE), CEIL(M,BLOCK_SIZE));  // tile according to C's shape (M rows, N cols)
sgemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= M || idy >= N) return;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Initialize the block tile's starting position
    A = &A[(by * BM) * K];
    B = &B[bx * BN];
    C = &C[(by * BM) * N + bx * BN];

    float accum = 0.0f;
    for (int k = 0; k < K; k += BK) {
        // move global ==> shared
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        __syncthreads();
        A = A + BK;
        B = B + BK * N;
        for (int i = 0; i < BK; i++) {
            accum += As[ty * BK + i] * Bs[i * BN + tx];
        }
        __syncthreads();
    }

    C[ty * N + tx] = accum;
}
```

## thread_tile

Each thread takes on more computation, making it more efficient:

```cpp
dim3 block(256);
dim3 grid(CEIL(N,128), CEIL(M,128));  // tile according to C's shape (M rows, N cols) // The grid shape is ONLY determined by the output shape.
sgemm<128, 128, 8, 8, 8><<<grid, block>>>(A,B,C,M,N,K);

template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdy.y;

    int block_row_thread = BN / TN;  // threads across one row of the block tile
    int block_col_thread = BM / TM;  // threads down one col of the block tile
    // thread_num is determined by the OUTPUT tile (BM×BN) and per-thread work (TM×TN),
    // NOT by the shared memory tile (BM×BK). BK is the K-loop step, independent of BN.
    // kernel3 (1D tile, TN=1): thread_num = BM*BN/TM
    // kernel4 (2D tile):       thread_num = BM*BN/(TM*TN)  ← fewer threads, more work per thread
    int thread_num = block_row_thread * block_col_thread;

    int tx = (threadIdx.x % block_row_thread) * TN;  // x coord of the thread tile's top-left
    int ty = (threadIdx.x / block_row_thread) * TM;  // y coord of the thread tile's top-left

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Pointer init: move each pointer to the top-left of this block's tile.
    // Row-major: element [row][col] = base[row * row_stride + col]
    //
    // A (M×K, row stride=K): row partition only, col starts at 0 (all K cols needed)
    //   A[by*BM][0] = A[by*BM * K + 0] = A[by*BM * K]
    A = &A[by * BM * K];
    //
    // B (K×N, row stride=N): col partition only, row starts at 0 (all K rows needed)
    //   B[0][bx*BN] = B[0 * N + bx*BN] = B[bx*BN]
    B = &B[bx * BN];
    //
    // C (M×N, row stride=N): both row and col partition, K does not appear (summed away)
    //   C[by*BM][bx*BN] = C[by*BM * N + bx*BN]
    C = &C[by * BM * N + bx * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;  // BM/(BM/(thread_num/BK)) = thread_num/BK = stride

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float accum[TM][TN] = {0.0f};
    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();

        // A[0][col + BK]
        A += BK;
        // B[row + BK][0]
        B += BK * N;

        for (int row = 0; row < TM; row++) {
            for (int col = 0; col < TN; col++) {
                for (int i = 0; i < BK; i++) {
                    // As[ty+row][i] * Bs[i][tx+col]
                    accum[row][col] += As[(ty + row) * BK + i] * Bs[i * BN + (tx + col)];
                }
            }
        }
        __syncthreads();
    }
    for (int row = 0; row < TM; row++) {
        for (int col = 0; col < TN; col++) {
            // C[ty+row][tx+col] = accum[row][col]
            C[(ty + row) * N + (tx + col)] = accum[row][col];
        }
    }
}
```

# gemv

**Importance**: **Low**

**Operator description**: computes a matrix multiplied by a vector. The approach is to have one warp per block, with each warp responsible for computing one row. It is recommended to study and understand this, because the pattern of using one warp to compute a row in gemv can be extended to a row-wise reduction over a matrix (a row-wise reduction over a 2D matrix, not just a 1D array).

**Source folder**: [./gemv](./gemv)

## gemv

```cpp
// number of rows: M = 1024
// number of cols: K = 32
// number of blocks equals number of rows: grid_size = M
// one warp per block: block_size = 32
sgemv<<<grid_size, block_size>>>(A, x, y, M, K);
__global__ void sgemv(float* A, float* x, float* y, int M, int K) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;  // 0~M-1
    if (row >= M) return;

    float res = 0.0f;
    int kIteration = CEIL(K, warpSize);  // number of elements each thread is responsible for

    for (int i = 0; i < kIteration; i++){
        int col = i * warpSize + laneId;
        res += (col < K) ? A[row * K + col] * x[col] : 0.0f;
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        // Folding the 'leave' lane into their 'parent' lanes
        // Each lane reads 'res' from landId + offset lane
        // No memory involved. Direct register to register transfer.
        // 0xFFFFFFFF means all 32 lanes are involved.
        res += __shfl_down_sync(0xFFFFFFFF, res, offset);
    }

    if(laneId == 0) y[row] = res;
}
```

## Extended application

Having understood gemv, we can follow the same idea to compute the softmax of each row of an MxN matrix. When M = 1, the problem becomes computing the softmax of an array of length N.

### softmax_matrix

Source: [./reduce/softmax_matrix/softmax_matrix.cu](./reduce/softmax_matrix/softmax_matrix.cu)

For an MxN matrix, computing the softmax of each row follows the same idea: each warp handles one row, using that warp to compute the row's sum and maximum, storing the results in shared memory, and then computing the softmax of each element:

```cpp
__global__ void softmax_kernel(float* input, float* output, int M, int N) {
    __shared__ float s_max_val;
    __shared__ float s_sum;
    int laneId = threadIdx.x % warpSize;
    // current row
    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);  // number of elements each thread is responsible for

    // find the max of each row
    float max_val = -FLT_MAX;
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    if (laneId == 0) s_max_val = max_val;  // the max is aggregated into the first thread, which moves it into s_mem

    // compute the sum of each row, subtracting the max
    float sum = 0.0f;
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - s_max_val) : 0.0f;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (laneId == 0) s_sum = sum;  // the sum is aggregated into the first thread, which moves it into s_mem

    // compute the softmax of each row
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        if (col < N) output[row * N + col] = expf(input[row * N + col] - s_max_val) / s_sum;
    }
}
```

After switching to `__shfl_xor_sync`, each thread's register holds the final `max_val` and `sum`, so there is no need to write to shared memory and read it back:

```cpp
dim3 block(32);
dim3 grid(M);

__global__ void softmax_kernel(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    // current row
    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);  // number of elements each thread is responsible for

    // find the max of each row
    float max_val = -FLT_MAX;
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        // __shfl_xor_sync reads the partner's value before it is overwritten.
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // compute the sum of each row, subtracting the max
    float sum = 0.0f;
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - max_val) : 0.0f;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // compute the softmax of each row
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        if (col < N) output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
    }
}
```

Furthermore, **when the number of rows M = 1, the problem degenerates into a reduction sum over an array of length N**. You can write this yourself.
```cpp
// Solution 1
dim3 block(32);
dim3 grid(1);
__global__ void reduce_sum(float* input, float* output, int N) {
    // use warpSize here because:
    // laneId is internal to a warp
    // But threadIdx.x is internal to the block, so we have a mismatch here.
    // To resolve the mismatch, we simply use 1-d block of size warpSize
    int laneId = threadIdx.x % warpSize;
    // warpSize is a constant
    int iterations = CEIL(N, warpSize);
    float sum = 0.0f
    for (int i = 0; i < iterations; i++) {
        int col = i * warpSize + laneId;
        sum += col < N ? input[col]; 0.0f;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>=1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    if (laneId == 0) {
        *output = sum;
    }
}

// Solution 2
dim3 block(32);
dim3 grid(CEIL(N,32));
__global__ void reduce_sum(float* input, float* output, int N) {
    int laneId = threadIdx.x % warpSize;
    __shared__ float shared_sum = 0.0f
    // warpSize is a constant

    // int iterations = CEIL(N, warpSize*grid_size); // wrong: grid_size is not built in.
    int iterations = CEIL(N, warpSize*gridDim.x);
    float sum = 0.0f
    for (int i = 0; i < iterations; i++) {
        int col = blockIdx.x * blockDim.x + i * warpSize + laneId;
        sum += input[col];
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>=1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    if (laneId == 0) {
        shared_sum += sum;
    }
    __syncthreads();
    if (blockIdx.x == 0) *output = shared_sum;
    // At this point we only have partial sum within each block. Because there is no inter-block communication
    // yet.
    // Option 1;
    atomicAdd(output, sum);
    // drawback: create contention from serializing results of each blocks.
    // Option 2:
    // launch a second kernel to reduce the partial sum.
    // float* partial;
    // NOTE: here we use HBM to achieve inter block communication.
    // cudaMalloc(&partial, grid_size * sizeof(float));  // one float per block
    // // stage 1: each block writes its partial sum to partial[blockIdx.x]
    // reduce_stage1<<<grid_size, block_size>>>(input, partial, N);
    // // stage 2: one block reduces all partial sums into final output
    // reduce_stage2<<<1, block_size>>>(partial, output, grid_size);
    // cudaFree(partial);

}
```
