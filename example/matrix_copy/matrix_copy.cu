#include <stdio.h>
#include <random>
#include <stdlib.h>

#define Ceil(a, b) ((a) + (b) - 1) / (b)

template <const int TILE_DIM>
__global__ void copy(const float *A, float *B, const int N) {
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = ny * N + nx;
    if (nx < N && ny < N) {
        B[index] = A[index];
    }
}

void randomize_matrix(float *mat, int N) {
    std::random_device rd;  
    std::mt19937 gen(rd()); // initialize the generator with the random device

    // create a uniform distribution over [0, 2000)
    std::uniform_int_distribution<> dis(0, 2000);
    for (int i = 0; i < N; i++) {
        // generate a random number, limited to the range [-1.0, 1.0]
        mat[i] = (dis(gen)-1000)/1000.0;  
    }
}


int main() {
    const int TILE_DIM = 32;
    const int N = 128;
    const dim3 grid_size(Ceil(N, TILE_DIM), Ceil(N, TILE_DIM));
    const dim3 block_size(TILE_DIM, TILE_DIM);

    // 1. init
    float *h_A = (float *)malloc(sizeof(float) * N * N);
    randomize_matrix(h_A, N * N);
    float *d_A, *d_B;
    cudaMalloc((void **) &d_A, sizeof(float) * N * N);
    cudaMalloc((void **) &d_B, sizeof(float) * N * N);
    cudaMemcpy(d_A, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    // 2. call copy
    copy<TILE_DIM><<<grid_size, block_size>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    return 0;
}
