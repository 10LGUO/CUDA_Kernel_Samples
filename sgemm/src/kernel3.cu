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
    // thread_num: threads per block. Each thread computes TM output elements,
    // so BM*BN total elements / TM elements per thread = BM*BN/TM threads needed.
    int thread_num = BM * BN / TM;

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move each pointer to the top-left corner of this block's tile.
    // All three matrices are row-major, so element [row][col] = base[row * row_stride + col].
    //
    // A (M×K, row stride=K):
    //   This block owns rows by*BM .. by*BM+BM-1, and ALL K columns (full K needed for dot product).
    //   → start at row by*BM, col 0  →  A[by*BM * K]
    //   bx does not appear: A is partitioned by row (by), not by column.
    //
    // B (K×N, row stride=N):
    //   This block owns cols bx*BN .. bx*BN+BN-1, and ALL K rows (full K needed for dot product).
    //   → start at row 0, col bx*BN  →  B[0*N + bx*BN] = B[bx*BN]
    //   by does not appear: ALL blocks share the same K rows of B — by only determines
    //   which rows of A and C are used, never which rows of B. Starting B at a non-zero
    //   row would skip part of K and produce a wrong (partial) dot product.
    //   The K loop advances B downward (B += BK*N) through all K rows.
    //
    // C (M×N, row stride=N):
    //   This block owns the BM×BN patch at row by*BM, col bx*BN.
    //   K does not appear: K is the reduction dimension, summed away — C has no K axis.
    //   → C[by*BM * N + bx*BN]
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // 当前线程负责搬运全局内存矩阵A的中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    // a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存
    // b_tile_* 同理

    // 若BM=64, BK=8, thread_num=512, 则a_tile_stride=64, a_tile_stride=BM，表示每个线程搬运 1 轮即可完成所需元素的搬运
    // 若BM=128, BK=8, thread_num=512, 则a_tile_stride=64, a_tile_stride=BM/2，表示每个线程搬运 2 轮即可完成所需元素的搬运
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM + 1] = {0.};  // 每个线程负责TM个元素，则需要申请TM个寄存器保存累加值，额外的一个寄存器用于缓存
#pragma unroll
    for (int k = 0; k < K; k += BK) {  // 窗口滑动
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        // 移动A,B指针到下一个矩阵块
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
            tmp[TM] = Bs[tx + i * BN];  // 额外的一个寄存器，避免反复从共享内存中读取Bs[tx + i * BN]
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
