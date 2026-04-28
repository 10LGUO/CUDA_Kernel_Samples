# Reduction - max

Reduce using warp shuffle. Because AtomicMax does not support the float type, it needs to be implemented by hand.

## Test

```log
[max_cpu]: total_time_h = 0.126157 ms
[max_kernel]: total_time_1 = 0.052634 ms
output = 12799.000000, output_ref = 12799.000000
```
