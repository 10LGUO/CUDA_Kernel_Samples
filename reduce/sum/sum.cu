#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

void host_reduce(float* x, const int N, float* sum) {
    *sum = 0.0;
    for (int i = 0; i < N; i++) {
        *sum += x[i];
    }
}

// reduce_v0: uses global memory
__global__ void device_reduce_v0(float* d_x, float* d_y) {
    const int tid = threadIdx.x;
    float *x = &d_x[blockIdx.x * blockDim.x];  // base address of the element block this block processes

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_y[blockIdx.x] = x[0];
    }
}

template <const int BLOCK_SIZE>
void call_reduce_v0(float* d_x, float* d_y, float* h_y, const int N, float* sum) {
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    device_reduce_v0<<<grid_size, block_size>>>(d_x, d_y);
    cudaMemcpy(h_y, d_y, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // a second reduction is needed on the host
    *sum = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        *sum += h_y[i];
    }
}

// reduce_v1: uses (static) shared memory
template <const int BLOCK_SIZE>
__global__ void device_reduce_v1(float* d_x, float* d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ float s_y[BLOCK_SIZE];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;  // move global mem to shared mem
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}

template <const int BLOCK_SIZE>
void call_reduce_v1(float* d_x, float* d_y, float* h_y, const int N, float* sum) {
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    device_reduce_v1<BLOCK_SIZE><<<grid_size, block_size>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // a second reduction is needed on the host
    *sum = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        *sum += h_y[i];
    }
}

// reduce_v2: uses (dynamic) shared memory
__global__ void device_reduce_v2(float* d_x, float* d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ float s_y[];  // dynamic shared memory
    s_y[tid] = (n < N) ? d_x[n] : 0.0;  // move global mem to shared mem
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}

template <const int BLOCK_SIZE>
void call_reduce_v2(float* d_x, float* d_y, float* h_y, const int N, float* sum) {
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    device_reduce_v2<<<grid_size, block_size, sizeof(float) * BLOCK_SIZE>>>(d_x, d_y, N);  // uses (dynamic) shared memory
    cudaMemcpy(h_y, d_y, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // a second reduction is needed on the host
    *sum = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        *sum += h_y[i];
    }
}

// reduce_v3: improved, introduces an atomic function so no second reduction on the CPU is needed
__global__ void device_reduce_v3(float* d_x, float* d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ float s_y[];  // dynamic shared memory
    s_y[tid] = (n < N) ? d_x[n] : 0.0;  // move global mem to shared mem
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);  // atomic function: reads *d_y, adds s_y[0], and writes back to address d_y
        // *d_y += s_y[0];  // wrong: if d_y is read by multiple threads simultaneously, the write-back result would be incorrect
    }
}

template <const int BLOCK_SIZE>
void call_reduce_v3(float* d_x, float* d_y, float* h_y, const int N) {
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *h_y = 0.0;  // zero out d_y on the host
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);  // copy to d_y
    device_reduce_v3<<<grid_size, block_size, sizeof(float) * BLOCK_SIZE>>>(d_x, d_y, N);  // uses (dynamic) shared memory
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);  // copy back to h_y
    cudaDeviceSynchronize();
}

// reduce_v4: warp shuffle reduction — avoids shared memory for intra-warp reduction
__global__ void device_reduce_v4(float* d_x, float* d_y, const int N) {
    // Shared memory sized for worst case: 1024 threads / 32 (warpSize) = 32 warps max.
    // Only one float per warp is stored here (the warp's partial sum).
    __shared__ float s_y[32];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;  // which warp this thread belongs to within the block
    int laneId = threadIdx.x % warpSize;  // position of this thread within its warp (0-31)

    // Load one element per thread from HBM into a register. Threads beyond N use 0 (identity for addition).
    float val = (idx < N) ? d_x[idx] : 0.0f;

    // Phase 1: intra-warp tree reduction using warp shuffles.
    // __shfl_down_sync(mask, val, offset): each lane receives val from lane+offset.
    // All 32 lanes execute in lockstep — no __syncthreads() needed within a warp.
    // After log2(32)=5 rounds, lane 0 holds the sum of all 32 lanes.
    // #pragma unroll: compile-time loop unrolling — replaces the loop with 5 explicit instructions.
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);  // lane i += lane (i + offset)
    }

    // Lane 0 of each warp writes its partial sum to shared memory.
    // s_y is the bridge between warps — shuffles cannot cross warp boundaries.
    if (laneId == 0) s_y[warpId] = val;

    // Ensure all warps have written their partial sums before warp 0 reads them.
    __syncthreads();

    // Phase 2: warp 0 reduces the per-warp partial sums stored in s_y.
    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;  // number of warps in this block = entries in s_y
        // Load one partial sum per lane; lanes beyond warpNum load 0.
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        // Same warp shuffle reduction as Phase 1 — lane 0 ends up with the block total.
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        // atomicAdd: safely accumulates this block's sum into the global output.
        // Required because multiple blocks write to the same d_y address concurrently.
        if (laneId == 0) atomicAdd(d_y, val);
    }
}

template <const int BLOCK_SIZE>
void call_reduce_v4(float* d_x, float* d_y, float* h_y, const int N) {
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *h_y = 0.0;  // zero out d_y on the host
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);  // copy to d_y
    device_reduce_v4<<<grid_size, block_size>>>(d_x, d_y, N);  // uses (dynamic) shared memory
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);  // copy back to h_y
    cudaDeviceSynchronize();
}


__global__ void device_reduce_v5(float* d_x, float* d_y, const int N) {
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
		if (laneId == 0) atomicAdd(d_y, val);
	}
}

template <const int BLOCK_SIZE>
void call_reduce_v5(float* d_x, float* d_y, float* h_y, const int N) {
    const int GRID_SIZE = CEIL(CEIL(N, BLOCK_SIZE), 4);  // divide by 4 here
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *h_y = 0.0;  // zero out d_y on the host
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);  // copy to d_y
    device_reduce_v5<<<grid_size, block_size>>>(d_x, d_y, N);  // uses (dynamic) shared memory
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);  // copy back to h_y
    cudaDeviceSynchronize();
}

int main() {
    size_t N = 100000000;
    constexpr size_t BLOCK_SIZE = 128;
    const int repeat_times = 10;

    // 1. host
    float *h_nums = (float *)malloc(sizeof(float) * N);
    float *sum = (float *)malloc(sizeof(float));
    randomize_matrix(h_nums, N);
    
    float total_time_h = TIME_RECORD(repeat_times, ([&]{host_reduce(h_nums, N, sum);}));
    // printf("init_matrix:\n");
    // print_matrix(h_nums, 1, N);
    printf("[reduce_host]: sum = %f, total_time_h = %f ms\n", *sum, total_time_h / repeat_times);

    // 2. device
    float *d_nums, *d_rd_nums;
    cudaMalloc((void **) &d_nums, sizeof(float) * N);
    cudaMalloc((void **) &d_rd_nums, sizeof(float) * CEIL(N, BLOCK_SIZE));
    float *h_rd_nums = (float *)malloc(sizeof(float) * CEIL(N, BLOCK_SIZE));
    
    // 2.1 call reduce_v0, global memory. Because reduce accumulates the result into d_nums (global memory), repeatedly running reduce_v0 makes the resulting sum grow larger and larger.
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_0 = TIME_RECORD(repeat_times, ([&]{call_reduce_v0<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v0]: sum = %f, total_time_0 = %f ms\n", *sum, total_time_0 / repeat_times);

    // 2.2 call reduce_v1, uses static shared memory; repeated runs do not affect sum
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_1 = TIME_RECORD(repeat_times, ([&]{call_reduce_v1<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v1]: sum = %f, total_time_1 = %f ms\n", *sum, total_time_1 / repeat_times);    

    // 2.3 call reduce_v2, changes v1 to dynamic shared memory, with unchanged performance
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_2 = TIME_RECORD(repeat_times, ([&]{call_reduce_v2<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v2]: sum = %f, total_time_2 = %f ms\n", *sum, total_time_2 / repeat_times);

    // 2.4 call reduce_v3, introduces an atomic function on top of v2, so no second reduction on the CPU is needed
    float *d_sum;
    cudaMalloc((void **) &d_sum, sizeof(float));
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_3 = TIME_RECORD(repeat_times, ([&]{call_reduce_v3<BLOCK_SIZE>(d_nums, d_sum, sum, N);}));
    printf("[reduce_v3]: sum = %f, total_time_3 = %f ms\n", *sum, total_time_3 / repeat_times);    

    // 2.5 call reduce_v4, uses warp shuffle
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_4 = TIME_RECORD(repeat_times, ([&]{call_reduce_v4<BLOCK_SIZE>(d_nums, d_sum, sum, N);}));
    printf("[reduce_v4]: sum = %f, total_time_4 = %f ms\n", *sum, total_time_4 / repeat_times);    

    // 2.6 call reduce_v5, uses warp shuffle + float4
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_5 = TIME_RECORD(repeat_times, ([&]{call_reduce_v5<BLOCK_SIZE>(d_nums, d_sum, sum, N);}));
    printf("[reduce_v5]: sum = %f, total_time_5 = %f ms\n", *sum, total_time_5 / repeat_times);    

    // free memory
    free(h_nums);
    free(sum);
    free(h_rd_nums);
    cudaFree(d_nums);
    cudaFree(d_rd_nums);
    cudaFree(d_sum);
    return 0;
}