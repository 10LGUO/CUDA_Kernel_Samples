# CUDA SGEMM Optimization

This project is adapted from [NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE), with more diagrams and CUDA programming explanations added. See the references at the end.

## Development environment
Device: NVIDIA GeForce GTX 1050
```
Device ID: 0
       *Number of SMs: 5
       Compute Capability Major: 6
       Compute Capability Minor: 1
       memoryBusWidth: 128
       *maxThreadsPerBlock: 1024
       maxThreadsPerMultiProcessor: 2048
       *totalGlobalMem: 2047M
       sharedMemPerBlock: 48KB
       *sharedMemPerMultiprocessor: 96KB
       totalConstMem: 64KB
       *multiProcessorCount: 5
       *Warp Size: 32
```

## Development workflow
1. Write kernel.cu under src.
2. Write the corresponding header under include, and include that header in include/kernel.cuh.
3. Call your kernel in the call_kernel function of src/utils.cu.
4. Build:
```bash
mkdir build && cd build
cmake ..
make
```
5. Run:
```bash
# run cuBLAS(0) or custom kernel(>0)
./main 0  # cuBLAS
./main 1  # kernel1
...
```
6. Test and plot:
```bash
pip install matplotlib
bash tools/test.sh  # logs are saved in ./test, images in ./images
```

## CUDA terminology
- **Memory access**: usually refers to the amount of data that a GPU core (CUDA core) or thread needs to read from or write to global memory.
- **Compute-to-memory-access ratio**: the ratio of compute per second to memory access per second.

## Kernel1: Naive implementation (global memory)
<div align=center>
<img src="./images/kernel_cublas_vs_1.png" width = "700"/>
</div>

### Code
```cpp
__global__ __launch_bounds__(1024) 
void sgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int id_x = blockIdx.x * blockDim.x + threadIdx.x; // id x
    int id_y = blockIdx.y * blockDim.y + threadIdx.y; // id y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[id_y * K + i] * B[i * N + id_x]; // two global memory accesses and one FMA (fused multiply-add)
    }
    C[id_y * N + id_x] = alpha * tmp + beta * C[id_y * N + id_x];
}
```

### Computation steps (illustrated)
Each logical thread is mapped to one element of matrix C; each thread is responsible for computing one element of C:
<div align=center>
<img src="./images/image.png" width = "500"/><img src="./images/image-1.png" width = "500"/>
</div>

### Analysis
Unoptimized matrix multiply has less than 1/10 the performance of cuBLAS. The detailed analysis is as follows:

1. **Low memory-access ratio**: each iteration performs one FMA (multiply-accumulate) and two global memory reads, for a compute-to-memory-access ratio of 1/2.
2. **High memory-access latency**: accessing **global memory** has **high latency**, taking hundreds of clock cycles.
3. **The low memory-access ratio cannot effectively hide the memory-access latency.**
4. Memory-access volume: computing each element of matrix C requires accessing 2K single-precision floats, so the full computation requires $2\times K\times M\times N$.
5. Elements at the same position are read repeatedly (computing elements in the same row of C shares the same row of A, and computing elements in the same column of C shares the same column of B).

> Dynamic global memory is allocated at runtime and is visible to **all threads** (and to the host as well). Use `cudaMalloc()` and `cudaFree()` to allocate and free it.

## Kernel2: From global memory to shared memory
<div align=center>
<img src="./images/kernel_1_vs_2.png" width = "500"/><img src="./images/kernel_cublas_vs_2.png" width = "500"/>
</div>

### Computation steps (illustrated)
<div align=center>
<img src="./images/describe_kernel_2.png" width = "400"/>
</div>
<div align=center>
<img src="./images/image-3.png" width = "800"/>
</div>

As shown on the left above, in matrix multiply, computing the result of each row of matrix C repeatedly reads the same row of matrix A (similarly, computing each column of matrix C repeatedly reads the same column of matrix B).

Exploiting this, we can partition A, B, and C into tiles of $BM\times BK$, $BK\times BN$, and $BM\times BN$. The three matrices form grids of $\frac{M}{BM}\times \frac{K}{BK}$, $\frac{K}{BK}\times \frac{N}{BN}$, and $\frac{M}{BM}\times \frac{N}{BN}$, as shown on the right above:
1. Allocate shared memory equal to the tile size in the block. Each block reads data from global memory and stores it in shared memory.
2. Since the tile size is larger than $1\times 1$, the number of global memory reads decreases in proportion to the tile size.
3. Because shared memory is shared within a block, when elements in a block repeatedly read the same row (column), they can read directly from shared memory.
4. Although the total number of reads increases (in the figure, global memory accesses become half of the original, while shared memory accesses equal the read count of the naive implementation), the number of global memory accesses drops significantly. And although there are many shared memory accesses, their latency is far lower than that of global memory, so the **total access latency still decreases significantly**.

### Analysis
Performance improves over kernel1, but there is still a large gap compared with cuBLAS. The detailed analysis is as follows:

1. **Memory-access volume drops significantly**: computing all elements of C requires reading $\frac{M}{BM}\times \frac{N}{BN} \times \frac{K}{BK} \times (BM \times BK+BK \times BN)=M \times N \times K \times (\frac{1}{BM}+ \frac{1}{BN})$ from global memory, which is $0.5 \times(\frac{1}{BM}+ \frac{1}{BN})$ of kernel1. The code uses BM=BN=32, so the memory-access volume becomes 1/32 of the original.
2. **The memory-access ratio does not change**: each computation still needs 2 memory-access instructions and 1 compute instruction.

> As the analysis shows, increasing the tile size (BM and BN) further reduces global memory accesses, but also increases shared memory usage. Since the shared memory per SM is fixed, allocating excessive shared memory in a single thread block limits the number of warps (for example, if the number of threads per block is fixed but the shared memory allocation per block increases, then the number of blocks assigned to an SM decreases, reducing the total number of threads and warps).

## Kernel3: Introducing a 1D thread tile and registers
<div align=center>
<img src="./images/kernel_2_vs_3.png" width = "500"/><img src="./images/kernel_cublas_vs_3.png" width = "500"/>
</div>

### Computation steps
> `pragma unroll` unrolls a loop (tells the compiler to duplicate the loop body multiple times). The benefit is eliminating loop overhead (such as loop index computation and loop termination checks).

<div align=center>
<img src="./images/image-4.png" width = "800"/>
</div>

1. Introduce a thread tile, i.e. one thread is responsible for computing multiple elements in the block. TM and TN denote the height and width of the thread tile respectively. In the figure above, TM=2, TN=1.
2. A 1D `tmp` of length TM+1 is defined, where `tmp[TM]` caches an element of Bs into a register, giving nearly zero access latency.
3. When TM=8, in the code below, each iteration of the outer loop accesses Bs once, then accesses As 8 times and computes 8 times, giving a compute-to-memory-access ratio of 8:9, much higher than the original 1:2.
```cpp
for (int i = 0; i < BK; i++) {
    tmp[TM] = Bs[tx + i * BN]; // an extra register, to avoid repeatedly reading Bs[tx + i * BN] from shared memory
    #pragma unroll  // loop unrolling, increasing instruction-level parallelism
    for (int j = 0; j < TM; j++) {  // 8 iterations
        tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
    }
}
```

### Analysis
1. Global memory-access volume: compared with the initial version, by caching a $64\times 64$ tile the memory-access volume drops to 1/64.
2. Compute-to-memory-access ratio: raised to 8:9, which can effectively hide memory-access latency.

## Kernel4: Introducing a 2D thread tile
<div align=center>
<img src="./images/kernel_3_vs_4.png" width = "500"/><img src="./images/kernel_cublas_vs_4.png" width = "500"/>
</div>

## Kernel5: Introducing a 2D thread tile, using registers to avoid repeated shared memory reads
<div align=center>
<img src="./images/kernel_4_vs_5.png" width = "500"/><img src="./images/kernel_cublas_vs_5.png" width = "500"/>
</div>

## Kernel6: Vectorized memory instruction FLOAT4 optimization
<div align=center>
<img src="./images/kernel_5_vs_6.png" width = "500"/><img src="./images/kernel_cublas_vs_6.png" width = "500"/>
</div>

### Computation steps
<div align=center>
<img src="./images/image6-1.png" width = "800"/>
</div>
Because copying is done in units of float4, to facilitate the subsequent multiplication (keeping data contiguous), As is transposed when A is copied into As; Bs is not:
<div align=center>
<img src="./images/image6-2.png" width = "800"/>
</div>
<div align=center>
<img src="./images/image6-3.png" width = "500"/>
</div>

### Analysis
Building on kernel5, we introduce the `float4` type. This is a CUDA extension type used for "vectorized access", copying 4 floats at a time as a group, reducing the number of memory instructions.

A few things to note when using vectorized access:
1. Types like float4 increase register pressure and reduce overall parallelism.
2. If the pointer is not aligned or the data type size is not a power of 2, vectorized access cannot be used.

> In almost all cases vectorized loads are preferable to scalar loads. Note however that using vectorized loads **increases register pressure** and **reduces overall parallelism**. So if you have a kernel that is already register limited or has very low parallelism, you may want to stick to scalar loads. Also, as discussed earlier, if your pointer is **not aligned** or your **data type size in bytes is not a power of two** you cannot use vectorized loads.

Vectorized access has the following benefits:
1. Increases bandwidth.
2. Reduces memory instructions (4 memory-copy instructions → 1 memory-copy instruction).
3. Reduces latency.

> Vectorized loads are a fundamental CUDA optimization that you should use when possible, because they **increase bandwidth**, reduce **instruction count**, and **reduce latency**. In this post, I've shown how you can easily incorporate vectorized loads into existing kernels with relatively few changes.

For more details, see the [official blog](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/).

## Kernel7: Data prefetching (double buffering)
<div align=center>
<img src="./images/kernel_6_vs_7.png" width = "500"/><img src="./images/kernel_cublas_vs_7.png" width = "500"/>
</div>

### Computation steps
Overall flowchart:
<div align=center>
<img src="./images/double-buffer.png" width = "800"/>
</div>

Here is the detailed flow.

1. Initialize data:

<div align=center>
<img src="./images/image7-1.png" width = "800"/>
</div>

2. First copy from global memory to shared memory:

<div align=center>
<img src="./images/image7-2.png" width = "800"/>
</div>

3. **Enter the loop**, global memory to shared memory:

<div align=center>
<img src="./images/image7-3.png" width = "800"/>
</div>

4. Compute, and copy the next round of data into registers:

<div align=center>
<img src="./images/image7-4.png" width = "800"/>
</div>

5. Copy the next round of data from registers into the other shared memory buffer:

<div align=center>
<img src="./images/image7-5.png" width = "800"/>
</div>

6. Finish this iteration and start the next one (back to step 3).

### Analysis
Do not overuse `__syncthreads()` in loops: overusing `__syncthreads()` can degrade performance because it prevents threads from executing in parallel. See [this article](https://blog.csdn.net/weixin_43844521/article/details/133945535).

> The following is excerpted and adapted from: https://blog.csdn.net/LostUnravel/article/details/138324342

On the GPU, **memory access and computation correspond to different hardware units**, and **these two units can execute in parallel**.

**Sequential execution of code** corresponds to the **order in which hardware instructions are issued** after compilation. Although instructions are issued in order, issuing is fast, and after an instruction is issued it takes some time to complete, corresponding to the clock cycles a given instruction needs to finish. Memory-access latency means that memory-access instructions have longer clock cycles than compute instructions.

In kernel 6, two `__syncthreads()` are needed: one after loading data from global memory into shared memory, and one after taking data from shared memory into registers and completing the computation. Therefore in kernel6, none of the block's threads can start computing until all of them have finished loading data into shared memory; likewise, the next round of data cannot be loaded into shared memory until all threads finish computing.

The double-buffered implementation of kernel7 has the following advantages:
1. **Block level**: shared memory is doubled — half is used to load the next round of data and half for the current computation. Before the loop begins, the first round of data is loaded; then inside the loop, the current already-loaded data is used for computation while the next round is loaded. This way the current round of computation does not wait for the data load, because the data it needs was already loaded in the previous round. This saves one `__syncthreads()`, so at the **block level** the GPU can issue later compute instructions early, hiding the memory-access latency of loading from global memory to shared memory.
2. **Thread level**: registers are also doubled, following the same idea as shared memory — half for loading the next round of data and half for the current computation. This way, at the **thread level**, the GPU can issue later compute instructions early, hiding the memory-access latency of loading from shared memory to registers.

Two more points to note:
1. The latency between the first data preparation and the first computation cannot be hidden, because the first computation depends on the first data load. Every subsequent computation loads the next round of data at the same time, so its access latency can be hidden.
2. This only saves the `__syncthreads()` between loading data and computing inside a block. But only after all threads of the current block finish computing can the window be moved (i.e. the tile introduced in kernel2 — only when the block's threads all finish computing can it move to the next tile). This `__syncthreads()` cannot be omitted.
3. The access-latency hiding at the thread level (register access) is not achieved through `__syncthreads()`, but simply by avoiding the dependency between computation and data access, letting the compiler issue compute instructions early.

# Q&A
## Why, during pre-fetch, does global memory need to go into a register first before being moved into shared memory?
Answer: Due to hardware limitations, before the Ampere architecture global memory and shared memory were not directly connected, so the transfer logic was to move to a register first, then to shared memory.

Even if the code directly assigns global memory to shared memory, the "write to register first, then to shared memory" step is included in the middle; the compiler just hides it.

## Kernel7's code still looks like it executes sequentially, and doesn't seem to achieve "overlap of data reads and computation"
Recommended reading: [CUDA study notes - GEMM optimization: double buffering (Prefetch) and resolving bank conflicts](https://blog.csdn.net/LostUnravel/article/details/138324342)

> At a glance, the code as a whole still looks like sequential-execution logic, and it seems it cannot achieve overlap, because it is still the "read one tile, write one tile" code pattern.
This is not actually the case. The key is to understand the instruction issuing and completion process the code corresponds to. On the GPU, memory access and computation correspond to different hardware units, and these two units can execute in parallel. **Sequential execution of code** corresponds to the **order in which hardware instructions are issued** after compilation. Although issuing is sequential, it is fast, and after an instruction is issued it takes some time to complete, corresponding to the clock cycles a given instruction needs to finish. Memory-access latency means that memory-access instructions have longer clock cycles than compute instructions.

# References
1. https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE
2. https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs
3. https://zhuanlan.zhihu.com/p/410278370
4. https://zhuanlan.zhihu.com/p/435908830
5. https://blog.csdn.net/u013013023/article/details/127245181
6. bank conflict: https://blog.csdn.net/xysjj/article/details/103885803
7. bank conflict: https://segmentfault.com/a/1190000007533157
8. vectorized loads: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
9. vectorized loads: https://www.zhihu.com/question/574968879/answer/3005751704
