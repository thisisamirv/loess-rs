# loess-rs

[![Crates.io](https://img.shields.io/crates/v/loess-rs.svg)](https://crates.io/crates/loess-rs)
[![Documentation](https://docs.rs/loess-rs/badge.svg)](https://docs.rs/loess-rs)
[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A high-performance implementation of LOESS (Locally Estimated Scatterplot Smoothing) in Rust. This crate provides a robust, production-ready implementation with support for confidence intervals, multiple kernel functions, and optimized execution modes.

> [!IMPORTANT]
> For parallelization or `ndarray` support, use [`fastLoess`](https://github.com/av746/fastLoess).

## Features

- **Robust Statistics**: IRLS with Bisquare, Huber, or Talwar weighting for outlier handling.
- **Multidimensional Smoothing**: Support for n-D data with customizable distance metrics (Euclidean, Manhattan, etc.).
- **Flexible Fitting**: Linear, Quadratic, Cubic, and Quartic local polynomials.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Delta optimization for skipping dense regions and streaming/online modes for large or real-time datasets.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.
- **Flexibility**: Multiple weight kernels (Tricube, Epanechnikov, etc.) and `no_std` support (requires `alloc`).
- **Validated**: Numerical agreement with R's `stats::loess` and Python's `statsmodels`.

## Robustness Advantages

This implementation is **more robust than statsmodels** due to two key design choices:

### MAD-Based Scale Estimation

For robustness weight calculations, this crate uses **Median Absolute Deviation (MAD)** for scale estimation:

```text
s = median(|r_i - median(r)|)
```

In contrast, statsmodels uses median of absolute residuals:

```text
s = median(|r_i|)
```

**Why MAD is more robust:**

- MAD is a **breakdown-point-optimal** estimator—it remains valid even when up to 50% of data are outliers.
- The median-centering step removes asymmetric bias from residual distributions.
- MAD provides consistent outlier detection regardless of whether residuals are centered around zero.

### Boundary Padding

This crate applies **boundary policies** (Extend, Reflect, Zero) at dataset edges:

- **Extend**: Repeats edge values to maintain local neighborhood size.
- **Reflect**: Mirrors data symmetrically around boundaries.
- **Zero**: Pads with zeros (useful for signal processing).

statsmodels does not apply boundary padding, which can lead to:

- Biased estimates near boundaries due to asymmetric local neighborhoods.
- Increased variance at the edges of the smoothed curve.

### Gaussian Consistency Factor

For interval estimation (confidence/prediction), residual scale is computed using:

```text
sigma = 1.4826 * MAD
```

The factor 1.4826 = 1/Phi^-1(3/4) ensures consistency with the standard deviation under Gaussian assumptions.

## Performance Advantages

Benchmarked against Python's `statsmodels`. Achieves **113-2813× faster performance** across all tested scenarios, with no regressions. Performance gains scale dramatically with dataset size.

### Summary

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Scalability**  | 5       | **481×**       | 1057×        |
| **Financial**    | 4       | **270×**       | 301×         |
| **Iterations**   | 6       | **238×**       | 248×         |
| **Pathological** | 4       | **234×**       | 220×         |
| **Scientific**   | 4       | **212×**       | 239×         |
| **Fraction**     | 6       | **218×**       | 268×         |
| **Genomic**      | 4       | **6.9×**       | 10.4×        |
| **Delta**        | 4       | **5.0×**       | 5.0×         |

### Top 10 Performance Wins

| Benchmark        | statsmodels | Rust   | Speedup   |
|------------------|-------------|--------|-----------|
| scale_100000     | 43.7s       | 15.5ms | **2813×** |
| scale_50000      | 11.2s       | 7.6ms  | **1466×** |
| fraction_0.05    | 197.2ms     | 0.38ms | **516×**  |
| financial_10000  | 497.1ms     | 0.97ms | **512×**  |
| scale_10000      | 663.1ms     | 1.38ms | **481×**  |
| scientific_10000 | 777.2ms     | 1.86ms | **418×**  |
| financial_5000   | 170.9ms     | 0.49ms | **346×**  |
| fraction_0.1     | 227.9ms     | 0.67ms | **339×**  |
| scale_5000       | 229.9ms     | 0.69ms | **334×**  |
| iterations_0     | 74.2ms      | 0.26ms | **289×**  |

Check [Benchmarks](https://github.com/thisisamirv/loess-rs/tree/bench/benchmarks) for detailed results and reproducible benchmarking code.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
loess = "0.1"
```

For `no_std` environments:

```toml
[dependencies]
loess = { version = "0.1", default-features = false }
```

## Quick Start

```rust
use loess::prelude::*;

fn main() -> Result<(), LoessError> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    // Basic smoothing
    let result = Loess::new()
        .fraction(0.5)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;

    println!("Smoothed values: {:?}", result.y);
    Ok(())
}
```

## Builder Methods

```rust
use loess::prelude::*;

Loess::new()
    // Smoothing span (0, 1]
    .fraction(0.5)

    // Polynomial degree (Constant, Linear, Quadratic, Cubic, Quartic)
    .degree(Linear)

    // Number of dimensions
    .dimensions(1)

    // Distance metric (Euclidean, Manhattan, etc.)
    .distance_metric(Euclidean)

    // Robustness iterations
    .iterations(3)

    // Interpolation threshold
    .delta(0.01)

    // Kernel selection
    .weight_function(Tricube)

    // Robustness method
    .robustness_method(Bisquare)

    // Zero-weight fallback behavior
    .zero_weight_fallback(UseLocalMean)

    // Boundary handling (for edge effects)
    .boundary_policy(Extend)

    // Confidence intervals
    .confidence_intervals(0.95)

    // Prediction intervals
    .prediction_intervals(0.95)

    // Diagnostics
    .return_diagnostics()
    .return_residuals()
    .return_robustness_weights()

    // Cross-validation (for parameter selection)
    .cross_validate(KFold(5, &[0.3, 0.5, 0.7]).seed(123))

    // Convergence
    .auto_converge(1e-4)

    // Execution mode
    .adapter(Batch)

    // Build the model
    .build()?;
```

### Result Structure

```rust
pub struct LoessResult<T> {
    // Sorted x values
    pub x: Vec<T>,

    // Smoothed y values
    pub y: Vec<T>,

    // Point-wise standard errors
    pub standard_errors: Option<Vec<T>>,

    // Confidence intervals
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,

    // Prediction intervals
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,

    // Residuals
    pub residuals: Option<Vec<T>>,

    // Final IRLS weights
    pub robustness_weights: Option<Vec<T>>,

    // Diagnostics
    pub diagnostics: Option<Diagnostics<T>>,

    // Actual iterations used
    pub iterations_used: Option<usize>,

    // Selected fraction
    pub fraction_used: T,

    // CV RMSE per fraction
    pub cv_scores: Option<Vec<T>>,
}
```

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
for chunk in data_chunks {
    let result = processor.process_chunk(&chunk.x, &chunk.y)?;
}

// Finalize processing
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
for (x, y) in data_stream {
    if let Some(output) = processor.add_point(x, y)? {
        println!("Smoothed: {}", output.smoothed);
    }
}
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Local, captures rapid changes (wiggly)
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)
- **Use CV** when uncertain

### Robustness Method

- **Bisquare** (default): Best all-around, smooth, efficient
- **Huber**: Theoretically optimal MSE

### Robustness Iterations

- **0**: Clean data, speed critical
- **1-2**: Light contamination
- **3**: Default, good balance (recommended)
- **4-5**: Heavy outliers
- **>5**: Diminishing returns

### Kernel Function

- **Tricube** (default): Best all-around, smooth, efficient
- **Epanechnikov**: Theoretically optimal MSE
- **Gaussian**: Very smooth, no compact support
- **Uniform**: Fastest, least smooth (moving average)

### Polynomial Degree

- **Constant**: Moving weighted average (smoothing only)
- **Linear** (default): Standard LOESS, good for most trends
- **Quadratic**: Captures peaks and valleys better
- **Cubic/Quartic**: Specialized high-order fitting

### Distance Metric

- **Euclidean** (default): Standard L2 distance
- **Manhattan**: L1 distance, robust to outliers in X
- **Chebyshev**: L-infinity (max component)
- **Normalized**: Standardized Euclidean (useful for mixed scales)

### Delta Optimization

- **None**: Small datasets (n < 1000)
- **0.01 × range(x)**: Good starting point for dense data
- **Manual tuning**: Adjust based on data density

## Examples

Check the `examples` directory for more complex scenarios:

```bash
cargo run --example batch_smoothing
cargo run --example online_smoothing
cargo run --example streaming_smoothing
```

## MSRV

Rust **1.85.0** or later (2024 Edition).

## Validation

Validated against:

- **Python (statsmodels)**: Passed on 44 distinct test scenarios.
- **Original Paper**: Reproduces Cleveland (1979) results.

Check [Validation](https://github.com/thisisamirv/loess-rs/tree/bench/validation) for more information. Small variations in results are expected due to differences in scale estimation and padding.

## Related Work

- [fastLoess (Rust)](https://github.com/thisisamirv/fastloess)
- [fastLoess (Python wrapper)](https://github.com/thisisamirv/fastloess-py)
- [fastLoess (R wrapper)](https://github.com/thisisamirv/fastloess-R)

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## License

Dual-licensed under **AGPL-3.0** (Open Source) or **Commercial License**.
Contact `<thisisamirv@gmail.com>` for commercial inquiries.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *Journal of the American Statistical Association*.
- Cleveland, W.S. & Devlin, S.J. (1988). "Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting". *Journal of the American Statistical Association*.
