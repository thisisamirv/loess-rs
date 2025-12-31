# loess-rs

[![Crates.io](https://img.shields.io/crates/v/loess-rs.svg)](https://crates.io/crates/loess-rs)
[![Documentation](https://docs.rs/loess-rs/badge.svg)](https://docs.rs/loess-rs)
[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A high-performance implementation of LOESS (Locally Estimated Scatterplot Smoothing) in Rust. This crate provides a robust, production-ready implementation with support for confidence intervals, multiple kernel functions, and optimized execution modes.

## LOESS vs. LOWESS

| Feature               | LOESS (This Crate)                | LOWESS                         |
|-----------------------|-----------------------------------|--------------------------------|
| **Polynomial Degree** | Linear, Quadratic, Cubic, Quartic | Linear (Degree 1)              |
| **Dimensions**        | Multivariate (n-D support)        | Univariate (1-D only)          |
| **Flexibility**       | High (Distance metrics)           | Standard                       |
| **Complexity**        | Higher (Matrix inversion)         | Lower (Weighted average/slope) |

> [!TIP]
> For a **LOWESS** implementation which is faster and simpler, use [`lowess`](https://github.com/thisisamirv/lowess).

## Features

- **Robust Statistics**: IRLS with Bisquare, Huber, or Talwar weighting for outlier handling.
- **Multidimensional Smoothing**: Support for n-D data with customizable distance metrics (Euclidean, Manhattan, etc.).
- **Flexible Fitting**: Linear, Quadratic, Cubic, and Quartic local polynomials.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Interpolation surface with Tensor Product Hermite interpolation and streaming/online modes for large or real-time datasets.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.
- **Flexibility**: Multiple weight kernels (Tricube, Epanechnikov, etc.) and `no_std` support (requires `alloc`).
- **Validated**: Numerical twin of R's `stats::loess` with exact match (< 1e-12 diff).

## Performance

Benchmarked against R's `loess`. Achieves **3.3×–25× faster performance** across all tested scenarios. No regressions observed.

### Summary

| Category               | Matched | Median Speedup | Mean Speedup |
|------------------------|---------|----------------|--------------|
| **Fraction**           | 6       | **6.03×**      | 9.30×        |
| **Iterations**         | 6       | **8.79×**      | 8.91×        |
| **Polynomial Degrees** | 2       | **8.84×**      | 8.84×        |
| **Pathological**       | 4       | **6.88×**      | 7.58×        |
| **Financial**          | 3       | **4.30×**      | 4.36×        |
| **Scalability**        | 2       | **3.99×**      | 3.99×        |
| **Dimensions**         | 3       | **3.85×**      | 3.91×        |
| **Scientific**         | 3       | **3.75×**      | 3.70×        |
| **Genomic**            | 2       | **3.32×**      | 3.32×        |

### Top 10 Performance Wins

| Benchmark        | Rust   | R       | Speedup    |
|------------------|--------|---------|------------|
| fraction_0.67    | 0.86ms | 21.63ms | **25.23×** |
| fraction_0.5     | 1.14ms | 12.85ms | **11.25×** |
| iterations_1     | 0.76ms | 8.44ms  | **11.12×** |
| high_noise       | 1.50ms | 15.86ms | **10.55×** |
| degree_quadratic | 0.79ms | 7.86ms  | **9.91×**  |
| iterations_2     | 0.92ms | 8.95ms  | **9.76×**  |
| iterations_3     | 1.08ms | 9.73ms  | **9.01×**  |
| iterations_5     | 1.49ms | 12.73ms | **8.57×**  |
| degree_linear    | 0.76ms | 5.86ms  | **7.76×**  |
| iterations_0     | 0.75ms | 5.69ms  | **7.56×**  |

Check [Benchmarks](https://github.com/thisisamirv/loess-rs/tree/bench/benchmarks) for detailed results and reproducible benchmarking code.

## Validation

The Rust `loess-rs` crate is a **numerical twin** of R's `loess` implementation:

| Aspect          | Status         | Details                                    |
|-----------------|----------------|--------------------------------------------|
| **Accuracy**    | ✅ EXACT MATCH | Max diff < 1e-12 across all scenarios      |
| **Consistency** | ✅ PERFECT     | 20/20 scenarios pass with strict tolerance |
| **Robustness**  | ✅ VERIFIED    | Robust smoothing matches R exactly         |

Check [Validation](https://github.com/thisisamirv/loess-rs/tree/bench/validation) for detailed scenario results.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
loess-rs = "0.1"
```

For `no_std` environments:

```toml
[dependencies]
loess-rs = { version = "0.1", default-features = false }
```

## Quick Start

```rust
use loess_rs::prelude::*;

fn main() -> Result<(), LoessError> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    // Build and fit model
    let result = Loess::new()
        .fraction(0.5)      // Use 50% of data for each local fit
        .iterations(3)      // 3 robustness iterations
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;

    println!("{}", result);
    Ok(())
}
```

```text
Summary:
  Data points: 5
  Fraction: 0.5

Smoothed Data:
       X     Y_smooth
  --------------------
    1.00     2.00000
    2.00     4.10000
    3.00     5.90000
    4.00     8.20000
    5.00     9.80000
```

## Builder Methods

All builder parameters have sensible defaults. You only need to specify what you want to change.

```rust
use loess_rs::prelude::*;

Loess::new()
    // Smoothing span (0, 1] - default: 0.67
    .fraction(0.5)

    // Polynomial degree - default: Linear
    .degree(Quadratic)

    // Number of dimensions - default: 1
    .dimensions(2)

    // Distance metric - default: Euclidean
    .distance_metric(Manhattan)

    // Robustness iterations - default: 3
    .iterations(5)

    // Kernel selection - default: Tricube
    .weight_function(Epanechnikov)

    // Robustness method - default: Bisquare
    .robustness_method(Huber)

    // Boundary handling - default: Extend
    .boundary_policy(Reflect)

    // Confidence intervals (Batch only)
    .confidence_intervals(0.95)

    // Prediction intervals (Batch only)
    .prediction_intervals(0.95)

    // Include diagnostics
    .return_diagnostics()
    .return_residuals()
    .return_robustness_weights()

    // Cross-validation (Batch only)
    .cross_validate(KFold(5, &[0.3, 0.5, 0.7]).seed(123))

    // Auto-convergence
    .auto_converge(1e-4)

    // Interpolation settings
    .surface_mode(Interpolation)

    // Interpolation cell size - default: 0.2
    .cell(0.2)

    // Execution mode
    .adapter(Batch)

    // Build the model
    .build()?;
```

### Execution Modes

| Adapter     | Use Case                          | Features                        |
|-------------|-----------------------------------|---------------------------------|
| `Batch`     | Complete datasets in memory       | All features supported          |
| `Streaming` | Large datasets (>100K points)     | Chunked processing, overlap     |
| `Online`    | Real-time data, sensor streams    | Incremental updates             |

## Streaming Processing

For datasets that don't fit in memory:

```rust
let mut processor = Loess::new()
    .fraction(0.3)
    .iterations(2)
    .adapter(Streaming)
    .chunk_size(1000)
    .overlap(100)
    .build()?;

// Process data in chunks
let result1 = processor.process_chunk(&chunk1_x, &chunk1_y)?;
let result2 = processor.process_chunk(&chunk2_x, &chunk2_y)?;

// Finalize to get remaining buffered data
let final_result = processor.finalize()?;
```

## Online Processing

For real-time data streams:

```rust
let mut processor = Loess::new()
    .fraction(0.2)
    .iterations(1)
    .adapter(Online)
    .window_capacity(100)
    .build()?;

// Process points as they arrive
for i in 1..=10 {
    let x = i as f64;
    let y = 2.0 * x + 1.0;
    if let Some(output) = processor.add_point(&[x], y)? {
        println!("Smoothed: {:.2}", output.smoothed);
    }
}
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Fine detail, may be noisy
- **0.3-0.5**: Moderate smoothing (good for most cases)
- **0.5-0.7**: Heavy smoothing, emphasizes trends
- **0.7-1.0**: Very smooth, may over-smooth
- **Default: 0.67** (Cleveland's choice)

### Robustness Iterations

- **0**: No robustness (fastest, sensitive to outliers)
- **1-3**: Light to moderate robustness (recommended)
- **4-6**: Strong robustness (for contaminated data)
- **7+**: Diminishing returns

### Polynomial Degree

- **Constant**: Local weighted mean (smoothing only)
- **Linear** (default): Standard LOESS, good bias-variance balance
- **Quadratic**: Better for peaks/valleys, higher variance
- **Cubic/Quartic**: Specialized high-order fitting

### Kernel Function

- **Tricube** (default): Best all-around, Cleveland's original choice
- **Epanechnikov**: Theoretically optimal MSE
- **Gaussian**: Maximum smoothness, no compact support
- **Uniform**: Fastest, least smooth (moving average)

### Boundary Policy

- **Extend** (default): Pad with constant values
- **Reflect**: Mirror data at boundaries (for periodic/symmetric data)
- **Zero**: Pad with zeros (signal processing)
- **NoBoundary**: Original Cleveland behavior

> **Note:** For nD data, `Extend` defaults to `NoBoundary` to preserve regression accuracy.

## Examples

```bash
cargo run --example batch_smoothing
cargo run --example online_smoothing
cargo run --example streaming_smoothing
```

## MSRV

Rust **1.85.0** or later (2024 Edition).

## Robustness Advantages

This implementation uses **MAD-based scale estimation** for robustness weight calculations:

```text
s = median(|r_i - median(r)|)
```

MAD is a **breakdown-point-optimal** estimator—it remains valid even when up to 50% of data are outliers, compared to the median of absolute residuals used by some other implementations.

Median Absolute Residual (MAR), which is the default Cleveland's choice, is also available through the `scaling_method` parameter.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Dual-licensed under **AGPL-3.0** (Open Source) or **Commercial License**.
Contact `<thisisamirv@gmail.com>` for commercial inquiries.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *Journal of the American Statistical Association*.
- Cleveland, W.S. & Devlin, S.J. (1988). "Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting". *Journal of the American Statistical Association*.
