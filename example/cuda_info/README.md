# View device info
## Build and run
Build and run:
```bash
nvcc main.cu -o main
./main
```
Output:
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

## Shared memory
When allocating shared memory with `__shared__` in a kernel, be mindful of sharedMemPerBlock (the shared-memory limit per block), for example:
```cpp
template<const int BLOCK_SIZE>
__global__ void sgemm_v2(...) {
    // allocate shared memory
    // NVIDIA GeForce GTX 1050's sharedMemPerBlock is 48KB=48*1024B=49152B(0xc000)
    // 1 float takes 4 Bytes, so (BM*BK + BK*BN) should <= 48*1024/4 = 12288
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
}
```
When too much shared memory is requested, the compiler throws an error:
```cpp
template __global__ void sgemm_v2<128>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
```
Error:
```
ptxas error   : Entry function '_Z8sgemm_v2ILi128EEviiifPfS0_fS0_' uses too much shared data (0x20000 bytes, 0xc000 max)
```
