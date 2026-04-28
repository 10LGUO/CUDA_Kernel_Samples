#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

void host_transpose(float* input, int M, int N, float* output) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = input[j * N + i];
        }
    }
}

// naive implementation
__global__ void device_transpose_v0(const float* input, float* output, int M, int N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

// coalesced writes
__global__ void device_transpose_v1(const float* input, float* output, int M, int N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < M) {
        output[row * M + col] = input[col * N + row];
    }
}

// explicitly call __ldg to reduce the impact of non-coalesced reads
__global__ void device_transpose_v2(const float* input, float* output, int M, int N) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < M) {
        output[row * M + col] = __ldg(&input[col * N + row]);
    }
}

// use shared memory as a staging area, coalescing reads + writes, but there is a bank conflict
template <const int TILE_DIM>
__global__ void device_transpose_v3(const float* input, float* output, int M, int N) {
    __shared__ float S[TILE_DIM][TILE_DIM];
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;

    if (y1 < M && x1 < N) {
        S[threadIdx.y][threadIdx.x] = input[y1 * N + x1];  // coalesced reads
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if (y2 < N && x2 < M) {
        // coalesced writes, but there is a bank conflict:
        // note that the 32 threads in a warp (32 consecutive threadIdx.x values)
        // map to shared-memory data with a stride of 32, i.e. these 32 threads happen to access
        // 32 data in the same bank, causing a 32-way bank conflict
        output[y2 * M + x2] = S[threadIdx.x][threadIdx.y];
    }
}

// use shared memory as a staging area, coalescing reads + writes, pad the shared memory to resolve the bank conflict
template <const int TILE_DIM>
__global__ void device_transpose_v4(const float* input, float* output, int M, int N) {
    __shared__ float S[TILE_DIM][TILE_DIM + 1];  // pad the shared memory to resolve the bank conflict
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;

    if (y1 < M && x1 < N) {
        S[threadIdx.y][threadIdx.x] = input[y1 * N + x1];  // coalesced reads
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if (y2 < N && x2 < M) {
        // after padding, the 32 threads in a warp (32 consecutive threadIdx.x values)
        // map to shared-memory data with a stride of 33
        // if the first thread accesses the first level of the first bank
        // then the second thread accesses the second level of the second bank
        // and so on: 32 threads access 32 different banks, so there is no bank conflict
        output[y2 * M + x2] = S[threadIdx.x][threadIdx.y];  // coalesced writes
    }
}

// use shared memory as a staging area, coalescing reads + writes, use swizzling to resolve the bank conflict
template <const int TILE_DIM>
__global__ void device_transpose_v5(const float* input, float* output, int M, int N) {
    __shared__ float S[TILE_DIM][TILE_DIM];  // no padding; use swizzling to resolve the bank conflict
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;

    if (y1 < M && x1 < N) {
        S[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y1 * N + x1];  // coalesced reads
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if (y2 < N && x2 < M) {
        // swizzling mainly uses the following two properties of XOR to avoid bank conflicts:
        // 1. closure of the operation  2. x1^y != x2^y if and only if x1 != x2
        // example:
        // row 1's access positions change from 0,0,0,0... to 0,1,2,3...
        // row 2's access positions change from 1,1,1,1... to 1,0,3,2...
        // row 3's access positions change from 2,2,2,2... to 2,3,0,1...
        // row 4's access positions change from 3,3,3,3... to 3,2,1,0...
        // this both fully utilizes the shared memory space (due to properties 1 and 2)
        // and ensures the threads in a warp never access the same bank (due to property 2)
        output[y2 * M + x2] = S[threadIdx.x][threadIdx.x ^ threadIdx.y];  // coalesced writes
    }
}

int main() {
    // input is M rows by N cols; after transpose it is N rows by M cols
    size_t M = 12800;
    size_t N = 1280;
    constexpr size_t BLOCK_SIZE = 32;
    const int repeat_times = 10;

    // -------------------- compute the transpose once on the host; the result is used for later verification -------------------- //
    float *h_matrix = (float *)malloc(sizeof(float) * M * N);
    float *h_matrix_tr_ref = (float *)malloc(sizeof(float) * N * M);
    randomize_matrix(h_matrix, M * N);
    host_transpose(h_matrix, M, N, h_matrix_tr_ref);
    // printf("init_matrix:\n");
    // print_matrix(h_matrix, M, N);
    // printf("host_transpose:\n");
    // print_matrix(h_matrix_tr_ref, N, M);

    float *d_matrix;
    cudaMalloc((void **) &d_matrix, sizeof(float) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    free(h_matrix);

    // --------------------------------call transpose_v0--------------------------------- //
    float *d_output0;
    cudaMalloc((void **) &d_output0, sizeof(float) * N * M);                              // device output memory
    float *h_output0 = (float *)malloc(sizeof(float) * N * M);                            // host memory, used to hold the device's output

    dim3 block_size0(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size0(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // tile according to input's shape (M rows, N cols)
    float total_time0 = TIME_RECORD(repeat_times, ([&]{device_transpose_v0<<<grid_size0, block_size0>>>(d_matrix, d_output0, M, N);}));
    cudaMemcpy(h_output0, d_output0, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output0, h_matrix_tr_ref, M * N);                                     // check correctness
    printf("[device_transpose_v0] Average time: (%f) ms\n", total_time0 / repeat_times);  // print the average time

    cudaFree(d_output0);
    free(h_output0);

    // --------------------------------call transpose_v1--------------------------------- //
    float *d_output1;
    cudaMalloc((void **) &d_output1, sizeof(float) * N * M);                              // device output memory
    float *h_output1 = (float *)malloc(sizeof(float) * N * M);                            // host memory, used to hold the device's output

    dim3 block_size1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size1(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));                            // tile according to output's shape (N rows, M cols)
    float total_time1 = TIME_RECORD(repeat_times, ([&]{device_transpose_v1<<<grid_size1, block_size1>>>(d_matrix, d_output1, M, N);}));
    cudaMemcpy(h_output1, d_output1, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output1, h_matrix_tr_ref, M * N);                                     // check correctness
    printf("[device_transpose_v1] Average time: (%f) ms\n", total_time1 / repeat_times);  // print the average time

    cudaFree(d_output1);
    free(h_output1);

    // --------------------------------call transpose_v2--------------------------------- //
    float *d_output2;
    cudaMalloc((void **) &d_output2, sizeof(float) * N * M);                              // device output memory
    float *h_output2 = (float *)malloc(sizeof(float) * N * M);                            // host memory, used to hold the device's output

    dim3 block_size2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size2(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));                            // tile according to output's shape (N rows, M cols)
    float total_time2 = TIME_RECORD(repeat_times, ([&]{device_transpose_v2<<<grid_size2, block_size2>>>(d_matrix, d_output2, M, N);}));
    cudaMemcpy(h_output2, d_output2, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output2, h_matrix_tr_ref, M * N);                                     // check correctness
    printf("[device_transpose_v2] Average time: (%f) ms\n", total_time2 / repeat_times);  // print the average time

    cudaFree(d_output2);
    free(h_output2);

    // --------------------------------call transpose_v3--------------------------------- //
    float *d_output3;
    cudaMalloc((void **) &d_output3, sizeof(float) * N * M);                              // device output memory
    float *h_output3 = (float *)malloc(sizeof(float) * N * M);                            // host memory, used to hold the device's output

    dim3 block_size3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size3(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // tile according to input's shape (M rows, N cols)
    float total_time3 = TIME_RECORD(repeat_times, ([&]{device_transpose_v3<BLOCK_SIZE><<<grid_size3, block_size3>>>(d_matrix, d_output3, M, N);}));
    cudaMemcpy(h_output3, d_output3, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output3, h_matrix_tr_ref, M * N);                                     // check correctness
    printf("[device_transpose_v3] Average time: (%f) ms\n", total_time3 / repeat_times);  // print the average time

    cudaFree(d_output3);
    free(h_output3);

    // --------------------------------call transpose_v4--------------------------------- //
    float *d_output4;
    cudaMalloc((void **) &d_output4, sizeof(float) * N * M);                              // device output memory
    float *h_output4 = (float *)malloc(sizeof(float) * N * M);                            // host memory, used to hold the device's output

    dim3 block_size4(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size4(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // tile according to input's shape (M rows, N cols)
    float total_time4 = TIME_RECORD(repeat_times, ([&]{device_transpose_v4<BLOCK_SIZE><<<grid_size4, block_size4>>>(d_matrix, d_output4, M, N);}));
    cudaMemcpy(h_output4, d_output4, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output4, h_matrix_tr_ref, M * N);
    printf("[device_transpose_v4] Average time: (%f) ms\n", total_time4 / repeat_times);

    cudaFree(d_output4);
    free(h_output4);

    // --------------------------------call transpose_v5--------------------------------- //
    float *d_output5;
    cudaMalloc((void **) &d_output5, sizeof(float) * N * M);                              // device output memory
    float *h_output5 = (float *)malloc(sizeof(float) * N * M);                            // host memory, used to hold the device's output

    dim3 block_size5(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size5(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // tile according to input's shape (M rows, N cols)
    float total_time5 = TIME_RECORD(repeat_times, ([&]{device_transpose_v5<BLOCK_SIZE><<<grid_size5, block_size5>>>(d_matrix, d_output5, M, N);}));
    cudaMemcpy(h_output5, d_output5, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output5, h_matrix_tr_ref, M * N);
    printf("[device_transpose_v5] Average time: (%f) ms\n", total_time5 / repeat_times);

    cudaFree(d_output5);
    free(h_output5);

    // ---------------------------------------------------------------------------------- //
    free(h_matrix_tr_ref);

    return 0;
}