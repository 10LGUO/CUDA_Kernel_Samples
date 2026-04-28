#include "kernel2.cuh"

template<const int BLOCK_SIZE>
__global__ void sgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // BM = rows of C this block owns (block tile height)
    // BN = cols of C this block owns (block tile width)
    // BK = K-dimension step per iteration (how many cols of A / rows of B loaded per loop)
    // Grid is sized (CEIL(N,BN), CEIL(M,BM)) — one block per BM×BN patch of C.
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    // blockId and threadId
    int bx = blockIdx.x;
    int by = blockIdx.y;    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // allocate shared memory
    // NVIDIA GeForce GTX 1050's sharedMemPerBlock is 48KB = 48*1024B = 49152B(0xc000)
    // 1 float takes 4 Bytes, so (BM*BK + BK*BN) should <= 48*1024/4 = 12288
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

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {  // sliding window
        // cache A_tile and B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // synchronize until all threads finish caching
        __syncthreads();  // sync threads within the same block to the same point
        // move the A, B pointers to the next matrix tile
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // the FMA computation reads the cached data; sync before writing the cache in the next round to ensure all threads finished computing
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}

// template instantiation declaration
template __global__ void sgemm_v2<16>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
template __global__ void sgemm_v2<32>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
template __global__ void sgemm_v2<64>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

// Note: pay attention to the `sharedMemPerBlock`,
// for example, when there is a template instantiation declaration like below:
// template __global__ void sgemm_v2<128>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
// compiler will throw error like below:
// ptxas error   : Entry function '_Z8sgemm_v2ILi128EEviiifPfS0_fS0_' uses too much shared data (0x20000 bytes, 0xc000 max)
