# Reduction - Matrix Softmax
Content: compute the softmax of **each row** of an MxN matrix.

Following [sgemv](../../gemv/sgemv_k32.cu), use one warp to handle the computation of one row (column).

Note that if you use `__shfl_down_sync`, the final result of the warp reduction is aggregated into the first thread, which must move this data into s_mem for the other threads in the same warp to use.

You can use `__shfl_xor_sync` instead, so that each thread's registers hold the final max_val and sum, and there is no need to write to shared memory and read it back.

Test, M = 2048, N = 64:
```
[softmax_row_cpu]: total_time_h = 2.601370 ms
[softmax_row_gpu]: total_time_d = 0.061936 ms
[softmax_col_cpu]: total_time_h = 5.155942 ms
[softmax_col_gpu]: total_time_d = 0.167795 ms
```