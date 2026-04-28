#include "kernel4.cuh"

template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm_v4(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;  // threads across one row of the block tile
    int block_col_thread = BM / TM;  // threads down one col of the block tile
    // thread_num: threads per block.
    // Each thread owns a TM×TN sub-tile of the BM×BN output patch.
    // Total output elements = BM*BN; per-thread work = TM*TN.
    // Threads needed = BM*BN / (TM*TN) = block_col_thread * block_row_thread.
    //
    // Compare with kernel3 (1D thread tile, TN=1): thread_num = BM*BN/TM.
    // Here TN>1 so fewer threads are needed for the same BM×BN tile.
    //
    // Note: thread_num is NOT BM*BK. BK is the K-loop step chosen for shared memory fit,
    // independent of BN. The thread count is always driven by the output tile and
    // per-thread work, not by the shared memory tile dimensions.
    int thread_num = block_row_thread * block_col_thread;

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // move to the current block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    The current thread moves element [a_tile_row][a_tile_col] of global memory into [a_tile_row][a_tile_col] of shared memory.
    a_tile_stride is the total number of rows the block's threads can move into shared memory per round.

    If BM=64, BK=8, thread_num=512, then a_tile_stride=64, a_tile_stride=BM, meaning each thread needs one round to move all its elements.
    If BM=128, BK=8, thread_num=512, then a_tile_stride=64, meaning each thread needs two rounds to move all its elements.
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.}; // each thread handles TM*TN elements, so it needs TM*TN registers to hold the accumulators, plus one extra register for caching
#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
#pragma unroll
            for (int j = 0; j < TM; j++) {
                for (int l = 0; l < TN; l++)
                    tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
    }
}

// template instantiation declaration
template __global__ void sgemm_v4<128, 128, 8, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
