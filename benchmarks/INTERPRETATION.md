# Benchmark Results

## Summary

The Rust `loess-rs` crate demonstrates consistent performance improvements over R's `loess` across all tested scenarios. Median speedups range from **4.4x to 21.6x** across different categories, with peak speedups reaching over **53x** in specific configurations. No regressions were observed.

## Category Comparison

| Category               | Matched | Median Speedup | Mean Speedup |
|------------------------|---------|----------------|--------------|
| **Polynomial Degrees** | 2       | **21.67x**     | 21.67x       |
| **Fraction**           | 6       | **13.36x**     | 19.70x       |
| **Iterations**         | 6       | **16.35x**     | 15.90x       |
| **Pathological**       | 4       | **16.14x**     | 16.27x       |
| **Dimensions**         | 3       | **7.98x**      | 8.01x        |
| **Scalability**        | 2       | **5.86x**      | 5.86x        |
| **Genomic**            | 2       | **5.27x**      | 5.27x        |
| **Financial**          | 3       | **4.88x**      | 5.97x        |
| **Scientific**         | 3       | **4.46x**      | 5.28x        |

## Top 10 Rust Wins

| Benchmark        | Rust   | R       | Speedup    |
|------------------|--------|---------|------------|
| fraction_0.67    | 0.83ms | 44.80ms | **53.71x** |
| fraction_0.5     | 1.22ms | 32.25ms | **26.50x** |
| degree_quadratic | 0.75ms | 19.21ms | **25.52x** |
| high_noise       | 1.60ms | 34.68ms | **21.68x** |
| iterations_1     | 0.80ms | 15.88ms | **19.94x** |
| iterations_0     | 0.75ms | 13.68ms | **18.25x** |
| degree_linear    | 0.78ms | 13.92ms | **17.81x** |
| clustered        | 1.15ms | 19.74ms | **17.17x** |
| iterations_2     | 0.97ms | 16.63ms | **17.13x** |
| iterations_3     | 1.13ms | 17.56ms | **15.57x** |

## Regressions

**None identified.** R was not faster than Rust in any of the matched benchmarks.

## Detailed Results

### Dimensions

| Name      | Rust   | R       | Speedup |
|-----------|--------|---------|---------|
| 1d_linear | 0.53ms | 4.23ms  | 7.98x   |
| 2d_linear | 1.69ms | 13.26ms | 7.86x   |
| 3d_linear | 3.49ms | 28.55ms | 8.18x   |

### Financial

| Name           | Rust   | R      | Speedup |
|----------------|--------|--------|---------|
| financial_1000 | 0.21ms | 1.04ms | 4.88x   |
| financial_500  | 0.12ms | 0.55ms | 4.43x   |
| financial_5000 | 0.98ms | 8.42ms | 8.60x   |

### Fraction

| Name          | Rust   | R       | Speedup |
|---------------|--------|---------|---------|
| fraction_0.05 | 1.40ms | 5.98ms  | 4.27x   |
| fraction_0.1  | 1.23ms | 8.58ms  | 6.99x   |
| fraction_0.2  | 1.07ms | 13.71ms | 12.76x  |
| fraction_0.3  | 1.41ms | 19.66ms | 13.97x  |
| fraction_0.5  | 1.22ms | 32.25ms | 26.50x  |
| fraction_0.67 | 0.83ms | 44.80ms | 53.71x  |

### Genomic

| Name         | Rust   | R      | Speedup |
|--------------|--------|--------|---------|
| genomic_1000 | 0.27ms | 0.93ms | 3.41x   |
| genomic_5000 | 1.16ms | 8.25ms | 7.12x   |

### Iterations

| Name          | Rust   | R       | Speedup |
|---------------|--------|---------|---------|
| iterations_0  | 0.75ms | 13.68ms | 18.25x  |
| iterations_1  | 0.80ms | 15.88ms | 19.94x  |
| iterations_10 | 2.10ms | 23.18ms | 11.06x  |
| iterations_2  | 0.97ms | 16.63ms | 17.13x  |
| iterations_3  | 1.13ms | 17.56ms | 15.57x  |
| iterations_5  | 1.43ms | 19.17ms | 13.43x  |

### Pathological

| Name             | Rust   | R       | Speedup |
|------------------|--------|---------|---------|
| clustered        | 1.15ms | 19.74ms | 17.17x  |
| constant_y       | 0.91ms | 13.72ms | 15.11x  |
| extreme_outliers | 2.10ms | 23.32ms | 11.11x  |
| high_noise       | 1.60ms | 34.68ms | 21.68x  |

### Polynomial Degrees

| Name             | Rust   | R       | Speedup |
|------------------|--------|---------|---------|
| degree_linear    | 0.78ms | 13.92ms | 17.81x  |
| degree_quadratic | 0.75ms | 19.21ms | 25.52x  |

### Scalability

| Name       | Rust   | R      | Speedup |
|------------|--------|--------|---------|
| scale_1000 | 0.26ms | 1.10ms | 4.20x   |
| scale_5000 | 1.16ms | 8.76ms | 7.52x   |

### Scientific

| Name            | Rust   | R       | Speedup |
|-----------------|--------|---------|---------|
| scientific_1000 | 0.32ms | 1.43ms  | 4.46x   |
| scientific_500  | 0.16ms | 0.59ms  | 3.65x   |
| scientific_5000 | 1.45ms | 11.18ms | 7.73x   |

## Notes

- Both use identical scenarios
- Rust crate: `loess-rs`
- R: `loess`
- Test date: 2026-01-05
