#include "kernel6.cuh"

template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm_v6(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread; // one thread computes TM*TN elements of the block

    // position, within the block, of the top-left element of this thread's thread tile
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];


    const int ldg_a_num = BK * BM / thread_num / 4; // each thread moves 4 floats, so moving everything into As takes ldg_a_num rounds per thread
    const int ldg_b_num = BK * BN / thread_num / 4; // each thread moves 4 floats, so moving everything into Bs takes ldg_b_num rounds per thread

    int a_tile_row = threadIdx.x / (BK / 4); // 4 floats per row form one memory block; this thread moves the a_tile_col-th block of row a_tile_row
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num; // BM rows total, moved over ldg_a_num rounds, a_tile_stride rows per round

    int b_tile_row = threadIdx.x / (BN / 4); // 4 floats per row form one memory block; this thread moves the b_tile_col-th block of row b_tile_row
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num; // BK rows total, moved over ldg_b_num rounds, b_tile_stride rows per round

    float accum[TM][TN] = {0.}; // each thread handles TM*TN elements, so it needs TM*TN registers to hold the accumulators

    // all parameters used to compute ldg_a_num must be const, otherwise they cannot be used to declare an array size
    float ldg_a_reg[4 * ldg_a_num] = {0.}; // each thread moves ldg_a_num rounds; registers cache ldg_a_num float4 elements, used to transpose the As matrix

    float a_frag[TM];  // cache for the As shared memory
    float b_frag[TN];  // cache for the Bs shared memory

    // move to the current block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;  // the ldg_index-th round
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            // store As transposed; ldg_a_reg is the intermediate cache so that reads can be done as FLOAT4
            As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // no transpose needed
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]); // offset to the current thread tile
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]); // offset to the current thread tile
            }
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            //float4 atmp = FETCH_FLOAT4(accum[m][n]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}

// template instantiation declaration
template __global__ void sgemm_v6<128, 128, 8, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
