/*
 * elementwise_add_float4
 *
 * Computes c[i] = a[i] + b[i] for all i in [0, N) using vectorized float4 access.
 *
 * Each thread handles 4 consecutive floats via a single 128-bit LDG.128 load
 * instruction (float4), reducing memory transactions 4x compared to scalar access.
 * This is a bandwidth-bound kernel — the bottleneck is HBM throughput, not compute.
 *
 * Thread mapping:
 *   thread t → elements [4t, 4t+1, 4t+2, 4t+3]
 *   idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4
 *
 * Grid/block sizing:
 *   block_size = 1024 threads
 *   grid_size  = CEIL(CEIL(N, 4), block_size)  // enough blocks to cover all elements
 *
 * Threads with idx >= N exit immediately (out-of-bounds guard for non-multiple-of-4 N).
 *
 * Scalar vs float4 comparison:
 *
 *   Without vectorization (1 float per thread):
 *     int idx = blockDim.x * blockIdx.x + threadIdx.x;
 *     c[idx] = a[idx] + b[idx];
 *     thread 0 → element 0, thread 1 → element 1, thread N → element N
 *
 *   With float4 vectorization (4 floats per thread):
 *     int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
 *     float4 tmp = FLOAT4(a[idx]);  // one 128-bit LDG.128 load
 *     thread 0 → elements [0,1,2,3], thread 1 → [4,5,6,7], thread N → [4N..4N+3]
 *
 *   float4 requires 4x fewer threads and 4x fewer memory instructions for the same N.
 */


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT4(a) *(float4*)(&(a))
#define CEIL(a,b) ((a+b-1)/(b))
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};


__global__ void elementwise_add_float4(float* a, float* b, float* c, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N) return;
    
    float4 tmp_a = FLOAT4(a[idx]);
    float4 tmp_b = FLOAT4(b[idx]);
    float4 tmp_c;
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;
    FLOAT4(c[idx]) = tmp_c;
}

int main() {
    constexpr int N = 7;
    float* a_h = (float*)malloc(N * sizeof(float));
    float* b_h = (float*)malloc(N * sizeof(float));
    float* c_h = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        a_h[i] = i;
        b_h[i] = N-1-i;
    }

    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;
    cudaCheck(cudaMalloc((void**)&a_d, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&b_d, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&c_d, N * sizeof(float)));
    // Copy data from host to device crossing PCIe bus.
    cudaCheck(cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b_d, b_h, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 1024;
    int grid_size = CEIL(CEIL(N,4), 1024);
    elementwise_add_float4<<<grid_size, block_size>>>(a_d, b_d, c_d, N);

    cudaCheck(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("a_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) printf("%f\n", a_h[i]);
        else printf("%f ", a_h[i]);
    }
    printf("b_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) printf("%f\n", b_h[i]);
        else printf("%f ", b_h[i]);
    }
    printf("c_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) printf("%f\n", c_h[i]);
        else printf("%f ", c_h[i]);
    }
    return 0;
}