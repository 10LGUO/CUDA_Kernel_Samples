#include "kernel7.cuh"

template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread; // one thread computes TM*TN elements of the block

    // position, within the block, of the top-left element of this thread's thread tile
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[2][BK * BM]; // double the shared memory size for buffering
    __shared__ float Bs[2][BK * BN];


    const int ldg_a_num = BK * BM / thread_num / 4; // each thread moves 4 floats, so moving everything into As takes ldg_a_num rounds across all threads
    const int ldg_b_num = BK * BN / thread_num / 4; // each thread moves 4 floats, so moving everything into Bs takes ldg_b_num rounds across all threads

    int a_tile_row = threadIdx.x / (BK / 4); // 4 floats per row form one memory block; this thread moves the a_tile_col-th block of row a_tile_row
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num; // BM rows total, moved over ldg_a_num rounds, a_tile_stride rows per round

    int b_tile_row = threadIdx.x / (BN / 4); // 4 floats per row form one memory block; this thread moves the b_tile_col-th block of row b_tile_row
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num; // BK rows total, moved over ldg_b_num rounds, b_tile_stride rows per round

    float accum[TM][TN] = {0.}; // each thread handles TM*TN elements, so it needs TM*TN registers to hold the accumulators, plus one extra register for caching

    // all parameters used to compute ldg_a_num must be const, otherwise they cannot be used to declare an array size
    float ldg_a_reg[4 * ldg_a_num] = {0.}; // each thread moves ldg_a_num rounds; registers cache ldg_a_num float4 elements, used to transpose the As matrix
    float ldg_b_reg[4 * ldg_b_num] = {0.}; // each thread moves ldg_b_num rounds; registers cache ldg_b_num float4 elements

    float a_frag[2][TM];  // cache for the As shared memory; double the register size for buffering
    float b_frag[2][TN];  // cache for the Bs shared memory; double the register size for buffering

    // move to the current block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // first global to shared
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
        int ldg_index = i / a_tile_stride * 4;  // the ldg_index-th round
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        // store As transposed; ldg_a_reg is the intermediate cache so that reads can be done as FLOAT4
        As[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
        As[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // no transpose needed
    }

    int write_index = 1;
    int load_index;
    int k = 0;
    do {  // enter the loop
        __syncthreads();  // sync once at the start of the loop
        // A += BK;
        // B += BK * N;
        // the sliding-window logic is folded directly into k in the A and B indices, so no separate advance is needed
        k += BK;
        // load global to reg
        if (k < K) {
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;  // the ldg_index-th round
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                        FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;  // the ldg_index-th round
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                        FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
            }
        }
        load_index = write_index ^ 1;
        // first shared to frag. Here, the accum[m][n] computation below must wait for the first shared-to-frag to finish before continuing.
        // In other words, this first load from shared memory to registers cannot hide the "shared-memory-to-register access latency".
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(
                        As[load_index][OFFSET(0, ty + m, BM)]); // offset to the current thread tile
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(
                        Bs[load_index][OFFSET(0, tx + n, BN)]); // offset to the current thread tile
            }
        // finished first shared to frag
#pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {  // computes BK-1 times; since it loads the next iteration's data, it can hide the "shared-memory-to-register access latency".
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(
                        As[load_index][OFFSET(bk + 1, ty + m, BM)]); // offset to the current thread tile
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(
                        Bs[load_index][OFFSET(bk + 1, tx + n, BN)]); // offset to the current thread tile
            }
#pragma unroll
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                }
            }
        }
#pragma unroll
        for (int m = 0; m < TM; m++) {  // only BK-1 were computed above; compute the BK-th here
#pragma unroll
            for (int n = 0; n < TN; n++) {
                accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
            }
        }
        // __syncthreads();  // no sync needed here, because As(Bs)[load_index] above and As(Bs)[write_index] below are separate memory
        if (k < K) {
            // load reg to shared
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;
                As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                        FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }

            write_index ^= 1;
        }
    } while (k < K);
    
    // C = alpha*AB+C
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}

// template instantiation declaration
template __global__ void sgemm_v7<128, 128, 8, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
