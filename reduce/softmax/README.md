# Reduction - Softmax
Content: compute the softmax of a given array.

Note that `__threadfence()` cannot achieve grid-level synchronization, so you cannot compute the sum and then divide by the sum within a single kernel, because mid-kernel there is no guarantee that the computed sum is the sum over all elements. The same applies to max. Therefore, in softmax.cu, computing max and sum each use a separate kernel, because separate kernels are guaranteed to synchronize all blocks in the grid:
```cpp
void call_softmax_v2(float* output, float* input_device, float* output_device, float* total_device, float* total_max_device, int N) {
    int block_size = BLOCK_SIZE;
    int grid_size  = CEIL(N, BLOCK_SIZE);

    // 1. initialize
    cudaCheck(cudaMemset(total_device, 0, sizeof(float)));  // total must be set to 0
    setToNegativeMax<<<1,1>>>(total_max_device);            // set total_max_device to the smallest FLOAT value

    // 2. compute the maximum
    max_kernel<<<grid_size, block_size>>>(input_device, total_max_device, N);

    // 3. compute the sum
    sum_kernel<<<grid_size, block_size>>>(input_device, total_device, total_max_device, N);

    // 4. compute softmax (subtract the max to avoid overflow)
    softmax_kernel<<<grid_size, block_size>>>(input_device, output_device, total_device, total_max_device, N);
}
```
