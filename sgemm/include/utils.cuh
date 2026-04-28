#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <random>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel.cuh"

// macro
#define REPEAT_TIMES 10
#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
// call `func` for `N` times and return total time(ms)
#define TIME_RECORD(N, func)                                                                    \
    [&] {                                                                                       \
        float total_time = 0;                                                                   \
        for (int repeat = 0; repeat <= N; ++repeat) {                                           \
            cudaEvent_t start, stop;                                                            \
            cudaCheck(cudaEventCreate(&start));                                                 \
            cudaCheck(cudaEventCreate(&stop));                                                  \
            cudaCheck(cudaEventRecord(start));                                                  \
            cudaEventQuery(start);                                                              \
            func();                                                                             \
            cudaCheck(cudaEventRecord(stop));                                                   \
            cudaCheck(cudaEventSynchronize(stop));                                              \
            float elapsed_time;                                                                 \
            cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));                        \
            if (repeat > 0) total_time += elapsed_time;                                         \
            cudaCheck(cudaEventDestroy(start));                                                 \
            cudaCheck(cudaEventDestroy(stop));                                                  \
        }                                                                                       \
        if (N == 0) return (float)0.0;                                                          \
        return total_time;                                                                      \
    }()

// CUDA
void _cudaCheck(cudaError_t error, const char *file, int line); // CUDA error check
void CudaDeviceInfo();                                         // print CUDA info

// matrix
void randomize_matrix(float *mat, size_t N);         // randomly initialize a matrix
void copy_matrix(float *src, float *dest, size_t N);    // copy a matrix
void print_matrix(const float *A, size_t M, size_t N);     // print a matrix
bool verify_matrix(float *mat1, float *mat2, size_t N); // verify a matrix

// call kernel
float call_kernel(int kernel_num, bool record, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
