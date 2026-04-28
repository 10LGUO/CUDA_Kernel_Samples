# Reduction - sum
Content: compute the sum of a given array.

## v0~v3
1. device_reduce_v0: uses **global memory** only, and N must be an integer multiple of BLOCK_SIZE.
2. device_reduce_v1: uses (static) **shared memory**; N no longer needs to be an integer multiple of BLOCK_SIZE, and the reduction does not modify the data in global memory.
3. device_reduce_v2: modified from v1 to use (dynamic) **shared memory**, with unchanged performance.
4. device_reduce_v3: modified from v2 to use an atomic function, no longer needing a second reduction on the CPU. **The drawback of v3 is that BLOCK_SIZE must be a power of 2, otherwise the halving operation computes incorrectly and produces large errors.**

## v4, v5
device_reduce_v4: computes using warp shuffle, reducing within a warp. **BLOCK_SIZE must be an integer multiple of 32, otherwise a warp with fewer than 32 threads is produced, which may access invalid data.**

Because threads within a warp are **naturally synchronized (hardware-level synchronization)**, there is no need to manually call `__syncthreads()`, giving better parallelism and higher efficiency.

device_reduce_v5: builds on v4 with float4 vectorized reads.

## Test
Test results for N=100000000, BLOCK_SIZE = 128:
```
[reduce_host]: sum = -1392.220947, total_time_h = 383.848877 ms
[reduce_v0]: sum = 12188807.000000, total_time_0 = 31.068247 ms
[reduce_v1]: sum = -1392.776123, total_time_1 = 19.648817 ms
[reduce_v2]: sum = -1392.776123, total_time_2 = 19.483204 ms
[reduce_v3]: sum = -1392.776123, total_time_3 = 15.859097 ms
[reduce_v4]: sum = -1392.792847, total_time_4 = 11.208912 ms
[reduce_v5]: sum = -1392.694214, total_time_5 = 4.105523 ms
```

## References:
1. Fundamentals and Practice of CUDA Programming (Fan Zheyong)