# Matrix Transpose

## How to optimize **global memory** access?
1. **Coalesce accesses as much as possible**, i.e. consecutive threads read consecutive memory, and try to make the base address of the accessed global memory a multiple of 32 bytes (the amount of data handled by one data transfer).
2. If you cannot coalesce both reads and writes simultaneously, you should **try to coalesce the writes**, because if the compiler can determine that a global memory variable is read-only inside the kernel, it will automatically use `__ldg()` to read global memory and cache the data, mitigating the impact of non-coalesced access. But this only works for reads; there is no equivalent for writes. Also, for the Kepler and Maxwell architectures, you need to **explicitly** use the `__ldg()` function, e.g. `B[ny * N + nx] = __ldg(&A[nx * N + ny])`.

Optimizing for global memory, the differences are as follows:
1. device_transpose_v0: reads are coalesced, writes are not.
2. device_transpose_v1: reads are not coalesced, writes are coalesced, giving a speedup.
3. device_transpose_v2: reads are not coalesced, writes are coalesced, and `__ldg` is used explicitly, giving a speedup (reason unclear; in principle v2 and v3 should have the same performance).

## How to use **shared memory** to optimize matrix transpose?

<div align=center>
<img src="./assets/sharedMem.png" width = "800"/>
</div>

Optimize transpose using shared memory:
1. device_transpose_v3: use shared memory as a staging area; both reads and writes are coalesced, but there is a **bank conflict**.
2. device_transpose_v4: **pad the shared memory** to resolve the bank conflict.
3. device_transpose_v5: use **swizzling** to resolve the bank conflict, without padding the shared memory.

## Runtime efficiency of the different kernels
Tested on GTX1050 with M = 12800, N = 1280, BLOCK_SIZE = 32:
```
[device_transpose_v0] Average time: (6.859354) ms
[device_transpose_v1] Average time: (4.310410) ms
[device_transpose_v2] Average time: (2.117488) ms
[device_transpose_v3] Average time: (3.805533) ms
[device_transpose_v4] Average time: (2.035469) ms
[device_transpose_v5] Average time: (2.023494) ms
```

## References
1. Fundamentals and Practice of CUDA Programming (Fan Zheyong)
2. [CUDA notes - coalesced memory access](https://zhuanlan.zhihu.com/p/641639133)
3. [CUDA memory access](https://zhuanlan.zhihu.com/p/632244210)
4. [CUDA: GPU implementation of matrix transpose (Shared Memory)](https://blog.csdn.net/m0_46197553/article/details/125646380)
5. [[CUDA study notes] optimizing the matrix transpose operator](https://blog.csdn.net/LostUnravel/article/details/137613493)
6. [GPU matrix transpose optimization (transpose)](https://blog.csdn.net/feng__shuai/article/details/114630831)
7. [CUDA-Shared-Memory-Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/#CUDA-Shared-Memory-Swizzling)
8. [On Bank Conflict and Swizzle](https://zhuanlan.zhihu.com/p/11132414477)