# Benchmark Results

## Summary

The Rust `loess-rs` crate demonstrates consistent performance improvements over R's `loess` across all tested scenarios. Median speedups range from **3.3x to 8.8x** across different categories, with peak speedups reaching over **25x** in specific configurations. No regressions were observed.

## Category Comparison

| Category               | Matched | Median Speedup | Mean Speedup |
|------------------------|---------|----------------|--------------|
| **Fraction**           | 6       | **6.03x**      | 9.30x        |
| **Iterations**         | 6       | **8.79x**      | 8.91x        |
| **Polynomial Degrees** | 2       | **8.84x**      | 8.84x        |
| **Pathological**       | 4       | **6.88x**      | 7.58x        |
| **Financial**          | 3       | **4.30x**      | 4.36x        |
| **Scalability**        | 2       | **3.99x**      | 3.99x        |
| **Dimensions**         | 3       | **3.85x**      | 3.91x        |
| **Scientific**         | 3       | **3.75x**      | 3.70x        |
| **Genomic**            | 2       | **3.32x**      | 3.32x        |

## Top 10 Rust Wins

| Benchmark        | Rust   | R       | Speedup    |
|------------------|--------|---------|------------|
| fraction_0.67    | 0.86ms | 21.63ms | **25.23x** |
| fraction_0.5     | 1.14ms | 12.85ms | **11.25x** |
| iterations_1     | 0.76ms | 8.44ms  | **11.12x** |
| high_noise       | 1.50ms | 15.86ms | **10.55x** |
| degree_quadratic | 0.79ms | 7.86ms  | **9.91x**  |
| iterations_2     | 0.92ms | 8.95ms  | **9.76x**  |
| iterations_3     | 1.08ms | 9.73ms  | **9.01x**  |
| iterations_5     | 1.49ms | 12.73ms | **8.57x**  |
| degree_linear    | 0.76ms | 5.86ms  | **7.76x**  |
| iterations_0     | 0.75ms | 5.69ms  | **7.56x**  |

## Regressions

**None identified.** R was not faster than Rust in any of the matched benchmarks.

## Detailed Results

### Dimensions

| Name      | Rust   | R       | Speedup |
|-----------|--------|---------|---------|
| 1d_linear | 0.54ms | 2.25ms  | 4.21x   |
| 2d_linear | 1.81ms | 6.66ms  | 3.68x   |
| 3d_linear | 3.42ms | 13.17ms | 3.85x   |

### Financial

| Name           | Rust   | R       | Speedup |
|----------------|--------|---------|---------|
| financial_1000 | 0.22ms | 0.91ms  | 4.23x   |
| financial_500  | 0.13ms | 0.57ms  | 4.54x   |
| financial_5000 | 1.02ms | 4.39ms  | 4.30x   |

### Fraction

| Name          | Rust   | R       | Speedup |
|---------------|--------|---------|---------|
| fraction_0.05 | 1.25ms | 4.42ms  | 3.54x   |
| fraction_0.1  | 1.15ms | 4.29ms  | 3.73x   |
| fraction_0.2  | 1.07ms | 5.97ms  | 5.59x   |
| fraction_0.3  | 1.34ms | 8.69ms  | 6.48x   |
| fraction_0.5  | 1.14ms | 12.85ms | 11.25x  |
| fraction_0.67 | 0.86ms | 21.63ms | 25.23x  |

### Genomic

| Name         | Rust   | R      | Speedup |
|--------------|--------|--------|---------|
| genomic_1000 | 0.26ms | 0.80ms | 3.12x   |
| genomic_5000 | 1.20ms | 4.23ms | 3.51x   |

### Iterations

| Name          | Rust   | R       | Speedup |
|---------------|--------|---------|---------|
| iterations_0  | 0.75ms | 5.69ms  | 7.56x   |
| iterations_1  | 0.76ms | 8.44ms  | 11.12x  |
| iterations_10 | 2.04ms | 15.15ms | 7.43x   |
| iterations_2  | 0.92ms | 8.95ms  | 9.76x   |
| iterations_3  | 1.08ms | 9.73ms  | 9.01x   |
| iterations_5  | 1.49ms | 12.73ms | 8.57x   |

### Pathological

| Name             | Rust   | R       | Speedup |
|------------------|--------|---------|---------|
| clustered        | 1.20ms | 7.86ms  | 6.54x   |
| constant_y       | 0.98ms | 5.88ms  | 6.01x   |
| extreme_outliers | 2.09ms | 15.11ms | 7.23x   |
| high_noise       | 1.50ms | 15.86ms | 10.55x  |

### Polynomial Degrees

| Name             | Rust   | R      | Speedup |
|------------------|--------|--------|---------|
| degree_linear    | 0.76ms | 5.86ms | 7.76x   |
| degree_quadratic | 0.79ms | 7.86ms | 9.91x   |

### Scalability

| Name       | Rust   | R      | Speedup |
|------------|--------|--------|---------|
| scale_1000 | 0.27ms | 1.00ms | 3.71x   |
| scale_5000 | 1.15ms | 4.91ms | 4.26x   |

### Scientific

| Name            | Rust   | R      | Speedup |
|-----------------|--------|--------|---------|
| scientific_1000 | 0.31ms | 1.12ms | 3.58x   |
| scientific_500  | 0.16ms | 0.60ms | 3.76x   |
| scientific_5000 | 1.45ms | 5.45ms | 3.75x   |

## Notes

- Both use identical scenarios
- Rust crate: `loess-rs`
- R: `loess`
- Test date: 2025-12-30
