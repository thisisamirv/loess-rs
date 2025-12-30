## Comparison with R and scikit-misc

A detailed investigation was conducted to understand why `scikit-misc` (Python) appears significantly faster than R's `loess()`, despite both using the same underlying Fortran code.

### The "60ms vs 1ms" Discrepancy

The initial observation that `scikit-misc` was nearly 50× faster than R for 5000 points was due to **unaligned benchmark configurations**:

- **R Benchmark**: Was explicitly using `surface="direct"`.
- **Python Benchmark**: Was using the default `surface="interpolate"`.

Direct fitting grows as $O(N^2)$ while interpolation is much closer to $O(N \log N)$.

### Aligned Comparison (n=5000)

When both implementations are aligned on the same parameters, the performance gap narrows significantly:

| Mode            | skmisc (Python/C) | R (Formula) | Rust (loess-rs) |
|-----------------|-------------------|-------------|-----------------|
| **Interpolate** | **1.23ms**        | 4.80ms      | 7.52ms          |
| **Direct**      | **53.2ms**        | 67.1ms      | N/A             |

#### Key Findings

1. **skmisc Efficiency**: even when aligned, `skmisc` is **~4× faster than R** in interpolation mode. This is likely due to the overhead of R's formula interface and complex object creation, whereas the Cython wrapper in `skmisc` has very low overhead.
2. **Direct Mode Parity**: In `direct` mode, the gap between `skmisc` and R is much smaller (~1.25×), confirming that the core numerical routines are identical and the difference is primarily interface overhead.
3. **Rust Interpolation**: Rust is currently slower than both in `interpolation` mode. This indicates that the current `InterpolationSurface` implementation in Rust (which performs a tree traversal for every query point) is less optimized than the original Fortran implementation.

### Recommendations

For maximum performance in Python, `scikit-misc` is the current leader due to its highly optimized C/Fortran core and thin wrapper. Rust's `direct` mode (not yet fully benchmarked against others) is expected to be competitive once the interpolation evaluation loop is optimized.
