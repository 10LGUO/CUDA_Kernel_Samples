#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include "utils.cuh"

// cpu: compute the softmax of each row
void softmax_row(float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        // row `row`
        float* input_tmp  = input + row * N;
        float* output_tmp = output + row * N;
        float max_val = *(std::max_element(input_tmp, input_tmp + N));  // compute the maximum of the input array
        float sum = 0;
        for (int i = 0; i < N; i++) {
            output_tmp[i] = std::exp(input_tmp[i] - max_val);  // subtract the max from each value before exp, to avoid overflow
            sum += output_tmp[i];
        }
        for (int i = 0; i < N; i++) {
            output_tmp[i] /= sum;
        }
    }
}

// cpu: compute the softmax of each column
void softmax_col(float* x, float* y, int M, int N) {
    for (int col = 0; col < N; col++) {
        // offset to the current column
        float* x_col = x + col;
        float* y_col = y + col;

        // compute the max and sum of the current column
        float max_val = -FLT_MAX;
        for (int i = 0; i < M; i++) {
            max_val = max(x_col[i*N], max_val);
        }
        float sum = 0;
        for (int i = 0; i < M; i++) {
            sum += exp(x_col[i*N] - max_val);
        }
        for (int i = 0; i < M; i++) {
            y_col[i*N] = exp(x_col[i*N] - max_val) / sum;
        }
    }
}

// gpu: compute the softmax of each row
__global__ void softmax_row_kernel(float* input, float* output, int M, int N) {
    __shared__ float s_max_val;
    __shared__ float s_sum;
    int laneId = threadIdx.x % warpSize;
    // current row
    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);  // number of elements each thread is responsible for

    // find the max of each row
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    if (laneId == 0) s_max_val = max_val;  // the max is aggregated into the first thread, which moves it into s_mem

    // compute the sum of each row, subtracting the max
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - s_max_val) : 0.0f;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (laneId == 0) s_sum = sum;  // the sum is aggregated into the first thread, which moves it into s_mem

    // compute the softmax of each row
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        if (col < N) output[row * N + col] = expf(input[row * N + col] - s_max_val) / s_sum;
    }
}

// gpu: compute the softmax of each row. After switching to __shfl_xor_sync, each thread's
// registers hold the final max_val and sum, so there is no need to write to shared memory and read it back
__global__ void softmax_row_kernel2(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    // current row
    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);  // number of elements each thread is responsible for

    // find the max of each row
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // compute the sum of each row, subtracting the max
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - max_val) : 0.0f;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // compute the softmax of each row
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        if (col < N) output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
    }
}

// gpu: compute the softmax of each column
__global__ void softmax_col_kernel(float* input, float* output, int M, int N) {
    __shared__ float s_max_val;
    __shared__ float s_sum;
    int laneId = threadIdx.x % warpSize;
    // current column
    int col = blockIdx.x;
    if (col >= N) return;

    int iteration = CEIL(M, warpSize);  // number of elements each thread is responsible for

    // find the max of each column
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int row = i * warpSize + laneId;
        max_val = (row < M) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    if (laneId == 0) s_max_val = max_val;  // the max is aggregated into the first thread, which moves it into s_mem

    // compute the sum of each column, subtracting the max
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int row = i * warpSize + laneId;
        sum += (row < M) ? expf(input[row * N + col] - s_max_val) : 0.0f;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (laneId == 0) s_sum = sum;  // the sum is aggregated into the first thread, which moves it into s_mem

    // compute the softmax of each column
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int row = i * warpSize + laneId;
        if (row < M) output[row * N + col] = expf(input[row * N + col] - s_max_val) / s_sum;
    }
}

// gpu: compute the softmax of each column. After switching to __shfl_xor_sync, each thread's
// registers hold the final max_val and sum, so there is no need to write to shared memory and read it back
__global__ void softmax_col_kernel2(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    // current column
    int col = blockIdx.x;
    if (col >= N) return;

    int iteration = CEIL(M, warpSize);  // number of elements each thread is responsible for

    // find the max of each column
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int row = i * warpSize + laneId;
        max_val = (row < M) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // compute the sum of each column, subtracting the max
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int row = i * warpSize + laneId;
        sum += (row < M) ? expf(input[row * N + col] - max_val) : 0.0f;
    }
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // compute the softmax of each column
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
        int row = i * warpSize + laneId;
        if (row < M) output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
    }
}


int main() {
    const int M = 2048;
    const int N = 64;
    const int repeat_times = 10;

    float* input      = (float*)malloc(M * N * sizeof(float));  // input is an M*N matrix
    float* output     = (float*)malloc(M * N * sizeof(float));  // output is an M*N matrix
    float* output_ref = (float*)malloc(M * N * sizeof(float));  // output is an M*N matrix (cpu)

    // initialize the input
    randomize_matrix(input, M*N);

    // cpu, compute the softmax of one row
    float total_time_h = TIME_RECORD(repeat_times, ([&]{softmax_row(input, output_ref, M, N);}));
    printf("[softmax_row_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

    float* input_device  = nullptr;
    float* output_device = nullptr;
    cudaCheck(cudaMalloc(&input_device,  M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&output_device, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(input_device, input, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // gpu, compute the softmax of one row
    float total_time_d = TIME_RECORD(repeat_times, ([&]{softmax_row_kernel2<<<M, 32>>>(input_device, output_device, M, N);}));
    printf("[softmax_row_gpu]: total_time_d = %f ms\n", total_time_d / repeat_times);
    cudaCheck(cudaMemcpy(output, output_device, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    verify_matrix(output, output_ref, M*N);

    // cpu, compute the softmax of one column
    float total_time_h2 = TIME_RECORD(repeat_times, ([&]{softmax_col(input, output_ref, M, N);}));
    printf("[softmax_col_cpu]: total_time_h = %f ms\n", total_time_h2 / repeat_times);

    // gpu, compute the softmax of one column
    float total_time_d2 = TIME_RECORD(repeat_times, ([&]{softmax_col_kernel2<<<N, 32>>>(input_device, output_device, M, N);}));
    printf("[softmax_col_gpu]: total_time_d = %f ms\n", total_time_d2 / repeat_times);
    cudaCheck(cudaMemcpy(output, output_device, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    verify_matrix(output, output_ref, M*N);

    free(input);
    free(output);
    free(output_ref);
    cudaCheck(cudaFree(input_device));
    cudaCheck(cudaFree(output_device));
    return 0;
}