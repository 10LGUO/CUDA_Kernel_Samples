#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include "utils.cuh"

const int N = 2048;
constexpr size_t BLOCK_SIZE = 256;
const int repeat_times = 10;

__global__ void setToNegativeMax(float* d_value) {
    *d_value = -FLT_MAX;
}

/*
 * atomicMax for floats — CUDA only provides integer atomics natively.
 *
 * Strategy: reinterpret float bits as int, use atomicCAS (Compare-And-Swap) in a
 * retry loop. IEEE 754 positive floats sort identically to their uint32 bit patterns,
 * so comparing bits as ints gives the correct max ordering.
 *
 * Example: *address=3.0, val=5.0
 *   old    = bits(3.0)
 *   assumed= bits(3.0)
 *   CAS writes bits(fmaxf(5.0, 3.0)) = bits(5.0) → succeeds, assumed==old, loop exits
 *
 * Race example: thread A (val=5.0) and thread B (val=7.0) both start with *address=3.0
 *   A wins CAS → *address = bits(5.0)
 *   B's CAS fails (assumed=bits(3.0) != current bits(5.0)) → retries
 *   B retries with assumed=bits(5.0) → writes bits(fmaxf(7.0,5.0))=bits(7.0) → succeeds
 *   Final: *address = 7.0 ✓
 *
 * static: limits visibility to this translation unit — avoids linker conflicts if
 * another .cu file defines its own atomicMax.
 */
__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;  // reinterpret float* as int* — same memory, same bits, different type
    int old = *address_as_i;            // read current bits of *address as int (e.g. bits(3.0f) = 0x40400000)
    int assumed;
    do {
        assumed = old;
        // atomicCAS(addr, expected, new):
        //   if *addr == expected → atomically write new, return expected
        //   if *addr != expected → do nothing, return current *addr
        // Either way, old is updated to the value in memory at time of CAS.
        // Values in *address only ever increase (fmaxf never writes smaller) so the loop
        // always terminates — each retry either succeeds or finds a value already >= our val.
        // __float_as_int / __int_as_float: pure bit reinterpret, no numeric conversion.
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);  // assumed != old means another thread wrote first — retry with updated old
    // Return value is the old value before our write. Callers typically discard it;
    // they only care that *address was updated, not what it previously held.
    return __int_as_float(old);
}

/*
 * max_kernel: finds the global maximum of input[0..N) using warp shuffle reduction.
 *
 * Example: input = [1.0, 9.0, 3.0, 7.0, ...], N=8, 1 block, 8 threads (1 warp)
 *
 * Phase 1 — intra-warp max via shuffle (no shared memory needed, runs in registers):
 *   Initial: lane0=1, lane1=9, lane2=3, lane3=7, lane4=2, lane5=6, lane6=4, lane7=8
 *   offset=4: lane0=max(1,2)=2, lane1=max(9,6)=9, lane2=max(3,4)=4, lane3=max(7,8)=8
 *   offset=2: lane0=max(2,4)=4, lane1=max(9,8)=9
 *   offset=1: lane0=max(4,9)=9  ← warp max
 *
 *   fmaxf(val, __shfl_down_sync(...)): each lane takes the max of itself and lane+offset.
 *   Threads beyond N load -FLT_MAX (identity for max) so they don't corrupt the result.
 *
 * Phase 2 — warp 0 reduces per-warp partial maxes stored in s_mem:
 *   (same shuffle pattern, but over s_mem values loaded into warp 0's lanes)
 *
 * Phase 3 — atomicMax writes block's max to global output:
 *   Multiple blocks race; atomicMax ensures the true global max survives.
 */
__global__ void max_kernel(float* input, float* output, int N) {
    __shared__ float s_mem[32];  // one slot per warp; max 1024 threads / 32 = 32 warps
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    // Load one element per thread; out-of-bounds threads use -FLT_MAX (identity for max)
    float val = (idx < N) ? input[idx] : (-FLT_MAX);

    // Phase 1: intra-warp max reduction via shuffle — no __syncthreads() needed (lockstep)
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));  // lane i = max(lane i, lane i+offset)
    }

    // Lane 0 of each warp writes its warp's max to shared memory
    if (laneId == 0) s_mem[warpId] = val;
    __syncthreads();  // ensure all warps have written before warp 0 reads

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        // Each lane of warp 0 loads one warp's partial max; unused lanes load -FLT_MAX
        val = (laneId < warpNum) ? s_mem[laneId] : (-FLT_MAX);

        // Phase 2: warp 0 reduces the per-warp maxes — lane 0 holds block's global max
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }

        // Phase 3: atomicMax accumulates across blocks — last block to write wins with true max
        if (laneId == 0) atomicMax(output, val);
    }
}


/*
 * sum_kernel: computes Σ e^(xi - max_val) over input[0..N) — the denominator of softmax.
 *
 * Subtracting max_val before exp() prevents float overflow (e.g. e^1000 = inf).
 * Mathematically equivalent: the max_val cancels out in softmax_kernel's division.
 *
 * Example: input=[1,2,3], max_val=3
 *   thread 0: val = e^(1-3) = e^-2 = 0.135
 *   thread 1: val = e^(2-3) = e^-1 = 0.368
 *   thread 2: val = e^(3-3) = e^0  = 1.0
 *   sum = 0.135 + 0.368 + 1.0 = 1.503
 *
 * Same warp shuffle + shared memory structure as max_kernel, but operator is += instead of fmaxf.
 * atomicAdd accumulates each block's partial sum into the global *sum.
 */
__global__ void sum_kernel(float* input, float* sum, float* max_val, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    // Compute e^(xi - M) per thread; out-of-bounds threads use 0 (identity for addition)
    float val = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;

    // Phase 1: intra-warp sum reduction via shuffle
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);  // lane i += lane i+offset
    }

    if (laneId == 0) s_mem[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : 0.0f;

        // Phase 2: warp 0 reduces per-warp partial sums
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // Phase 3: atomicAdd accumulates this block's sum into global *sum
        if (laneId == 0) atomicAdd(sum, val);
    }
}


/*
 * softmax_kernel: final normalization pass — one thread per element.
 *
 * Formula: output[i] = e^(input[i] - M) / sum
 *   where M = global max (from max_kernel), sum = Σ e^(xj-M) (from sum_kernel)
 *
 * Example (continuing from sum_kernel): input=[1,2,3], M=3, sum=1.503
 *   thread 0: output[0] = e^(1-3) / 1.503 = 0.135 / 1.503 = 0.090
 *   thread 1: output[1] = e^(2-3) / 1.503 = 0.368 / 1.503 = 0.245
 *   thread 2: output[2] = e^(3-3) / 1.503 = 1.0   / 1.503 = 0.665
 *   sum of outputs = 0.090 + 0.245 + 0.665 = 1.0 ✓
 *
 * No reduction needed — each thread independently normalizes its own element.
 * max_val and sum are single floats in HBM written by the previous two kernel launches.
 */
__global__ void softmax_kernel(float* input, float* output, float* sum, float* max_val, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) output[idx] = expf(input[idx] - *max_val) / (*sum);  // normalize by global sum
}

void softmax(float* input, float* output, int N, float* M, float* sum) {
    *M = *(std::max_element(input, input + N));  // 计算输入数组的最大值
    *sum = 0;
    for (int i = 0; i < N; i++) {
        output[i] = std::exp(input[i] - *M);  // 每个数先减去最大值，再求exp，避免溢出
        *sum += output[i];
    }
    for (int i = 0; i < N; i++) {
        output[i] /= *sum;
    }
}


void call_softmax_v1(float* output, float* input_device, float* output_device, float* total_device, float* total_max_device, int N) {
    int block_size = BLOCK_SIZE;
    int grid_size  = CEIL(N, BLOCK_SIZE);

    // 1. 初始化
    cudaCheck(cudaMemset(total_device, 0, sizeof(float)));      // total需要设置为0
    cudaCheck(cudaMemset(total_max_device, 0, sizeof(float)));
    
    // 2. 计算和
    sum_kernel<<<grid_size, block_size>>>(input_device, total_device, total_max_device, N);

    // 3. 计算softmax (没有减去最大值)
    softmax_kernel<<<grid_size, block_size>>>(input_device, output_device, total_device, total_max_device, N);
}


void call_softmax_v2(float* output, float* input_device, float* output_device, float* total_device, float* total_max_device, int N) {
    int block_size = BLOCK_SIZE;
    int grid_size  = CEIL(N, BLOCK_SIZE);

    // 1. 初始化
    cudaCheck(cudaMemset(total_device, 0, sizeof(float)));  // total需要设置为0
    setToNegativeMax<<<1,1>>>(total_max_device);            // total_max_device设置为最小FLOAT值

    // 2. 计算最大值
    max_kernel<<<grid_size, block_size>>>(input_device, total_max_device, N);

    // 3. 计算和
    sum_kernel<<<grid_size, block_size>>>(input_device, total_device, total_max_device, N);

    // 4. 计算softmax (减去最大值避免溢出)
    softmax_kernel<<<grid_size, block_size>>>(input_device, output_device, total_device, total_max_device, N);
}


int main() {
    float* input  = (float*)malloc(sizeof(float) * N);
    float* output_ref = (float*)malloc(sizeof(float) * N);
    float* M = (float*)malloc(sizeof(float));
    float* sum = (float*)malloc(sizeof(float));
    for (int i = 0; i < N; i++) {
        input[i] = i/(float)N;
    }
    float total_time_h = TIME_RECORD(repeat_times, ([&]{softmax(input, output_ref, N, M, sum);}));
    printf("[softmax_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

    float* input_device  = nullptr;
    float* output_device = nullptr;
    float* total_device = nullptr;
    float* total_max_device = nullptr;
    cudaCheck(cudaMalloc(&input_device, N * sizeof(float)));
    cudaCheck(cudaMalloc(&output_device, N * sizeof(float)));
    cudaCheck(cudaMalloc(&total_device, 1 * sizeof(float)));
    cudaCheck(cudaMalloc(&total_max_device, 1 * sizeof(float)));

    cudaCheck(cudaMemcpy(input_device, input, N * sizeof(float), cudaMemcpyHostToDevice));
    float* output = (float*)malloc(sizeof(float) * N);

    // softmax_v1
    float total_time_1 = TIME_RECORD(repeat_times, ([&]{call_softmax_v1(output, input_device, output_device, total_device, total_max_device, N);}));
    printf("[softmax_kernel1]: total_time_1 = %f ms\n", total_time_1 / repeat_times);
    cudaCheck(cudaMemcpy(output, output_device, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); 
    verify_matrix(output, output_ref, N);

    // softmax_v2
    float total_time_2 = TIME_RECORD(repeat_times, ([&]{call_softmax_v2(output, input_device, output_device, total_device, total_max_device, N);}));
    printf("[softmax_kernel2]: total_time_2 = %f ms\n", total_time_2 / repeat_times);
    cudaCheck(cudaMemcpy(output, output_device, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    verify_matrix(output, output_ref, N);

    float* total_host = (float*)malloc(sizeof(float));
    float* total_max_host = (float*)malloc(sizeof(float));
    cudaCheck(cudaMemcpy(total_host, total_device, sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(total_max_host, total_max_device, sizeof(float), cudaMemcpyDeviceToHost));

    free(input);
    free(output);
    free(M);
    free(sum);
    free(output_ref);
    cudaCheck(cudaFree(input_device));
    cudaCheck(cudaFree(output_device));
    cudaCheck(cudaFree(total_device));
    cudaCheck(cudaFree(total_max_device));
    return 0;
}