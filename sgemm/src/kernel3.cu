#include "kernel3.cuh"

// BM = rows of C this block owns (block tile height)
// BN = cols of C this block owns (block tile width)
// BK = K-dimension step per loop iteration
// TM = rows of C each thread owns (thread tile height)
//
// Grid: (CEIL(N,BN), CEIL(M,BM)) — one block per BM×BN patch of C
// Block: BM*BN/TM threads — fewer threads than elements because each thread handles TM rows
//
// Ownership hierarchy:
//   Block  → BM×BN patch of C
//   Thread → TM×1  column within the block's patch (1 col, TM rows)
template<const int BM,
         const int BN,
         const int BK,
         const int TM>
__global__ void sgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;  // which block-column: selects cols bx*BN..bx*BN+BN-1 of C (and B)
    int by = blockIdx.y;  // which block-row:    selects rows by*BM..by*BM+BM-1 of C (and A)
                          // grid.x = CEIL(N,BN), grid.y = CEIL(M,BM) — one block per BM×BN patch of C
                          // row coverage of A is split across grid.y blocks; K loop covers the col axis
    // thread_num: threads per block.
    // Each thread owns a TM×1 sub-tile (TM rows, 1 col) — this is a 1D thread tile.
    // Total output elements in the block tile = BM*BN.
    // Threads needed = BM*BN / TM.
    //
    // Note: this is NOT BM*BK. BK is the K-loop step (shared memory fit), independent of BN.
    // BM*BK threads would only accidentally equal BM*BN/TM for specific parameter values.
    // The thread count is always determined by the output tile (BM×BN) and per-thread work (TM),
    // not by the shared memory tile size.
    int thread_num = BM * BN / TM;

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Pointer init: move each pointer to the top-left of this block's tile.
    // Row-major: element [row][col] = base[row * row_stride + col]
    //
    // A (M×K, row stride=K): row partition only, col starts at 0 (all K cols needed)
    //   A[by*BM][0] = A[by*BM * K + 0] = A[by*BM * K]
    //   bx absent: A is not partitioned by column.
    A = &A[by * BM * K];
    //
    // B (K×N, row stride=N): col partition only, row starts at 0 (all K rows needed)
    //   B[0][bx*BN] = B[0 * N + bx*BN] = B[bx*BN]
    //   by absent: all blocks need the same K rows of B — starting at non-zero row
    //   would skip part of K and produce a wrong dot product.
    //   K loop advances B downward (B += BK*N) through all K rows.
    B = &B[bx * BN];
    //
    // C (M×N, row stride=N): both row and col partition, K absent (summed away, no K axis in C)
    //   C[by*BM][bx*BN] = C[by*BM * N + bx*BN]
    C = &C[by * BM * N + bx * BN];

    // The current thread moves element [a_tile_row][a_tile_col] of global memory matrix A into [a_tile_row][a_tile_col] of shared memory.
    // a_tile_stride is the number of rows the block's threads can move into shared memory per round.
    // b_tile_* is analogous.

    // If BM=64, BK=8, thread_num=512, then a_tile_stride=64, a_tile_stride=BM, meaning each thread needs 1 round to move all its elements.
    // If BM=128, BK=8, thread_num=512, then a_tile_stride=64, a_tile_stride=BM/2, meaning each thread needs 2 rounds to move all its elements.
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM + 1] = {0.};  // each thread handles TM elements, so it needs TM registers to hold the accumulators, plus one extra register for caching
#pragma unroll
    for (int k = 0; k < K; k += BK) {  // sliding window
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        // move the A, B pointers to the next matrix tile
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
            tmp[TM] = Bs[tx + i * BN];  // an extra register, to avoid repeatedly reading Bs[tx + i * BN] from shared memory
#pragma unroll
            for (int j = 0; j < TM; j++) {
                tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++) {
        C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
    }
}

// template instantiation declaration
template __global__ void sgemm_v3<64, 64, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
