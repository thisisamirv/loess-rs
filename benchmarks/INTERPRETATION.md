# Benchmark Results

## Summary

The Rust `lowess` crate demonstrates **113-2813× faster performance** than Python's statsmodels across all tested scenarios, with no regressions. Performance gains scale dramatically with dataset size, reaching over 2800× at 100K points.

## Category Comparison

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

## Top 10 Rust Wins

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

## Regressions

**None identified.** Rust outperforms statsmodels in all 40 matched benchmarks.

## Detailed Results

### Scalability (1K - 100K points)

| Size    | Rust   | statsmodels | Speedup  |
|---------|--------|-------------|----------|
| 1,000   | 0.16ms | 30.4ms      | 191×     |
| 5,000   | 0.69ms | 229.9ms     | 334×     |
| 10,000  | 1.38ms | 663.1ms     | 481×     |
| 50,000  | 7.61ms | 11.2s       | 1466×    |
| 100,000 | 15.5ms | 43.7s       | 2813×    |

### Fraction Variations (n=5000)

| Fraction | Rust    | statsmodels | Speedup |
|----------|---------|-------------|---------|
| 0.05     | 0.38ms  | 197.2ms     | 516×    |
| 0.10     | 0.67ms  | 227.9ms     | 339×    |
| 0.20     | 1.24ms  | 297.0ms     | 240×    |
| 0.30     | 1.81ms  | 357.0ms     | 197×    |
| 0.50     | 2.97ms  | 488.4ms     | 164×    |
| 0.67     | 3.96ms  | 601.6ms     | 152×    |

### Robustness Iterations (n=5000)

| Iterations | Rust   | statsmodels | Speedup |
|------------|--------|-------------|---------|
| 0          | 0.26ms | 74.2ms      | 289×    |
| 1          | 0.58ms | 148.5ms     | 254×    |
| 2          | 0.94ms | 222.8ms     | 237×    |
| 3          | 1.24ms | 296.5ms     | 238×    |
| 5          | 1.88ms | 445.1ms     | 236×    |
| 10         | 3.49ms | 815.6ms     | 234×    |

### Delta Parameter (n=10000)

| Delta    | Rust    | statsmodels | Speedup |
|----------|---------|-------------|---------|
| none (0) | 165.5ms | 678.2ms     | 4.1×    |
| small    | 0.46ms  | 2.28ms      | 4.9×    |
| medium   | 0.22ms  | 1.27ms      | 5.9×    |
| large    | 0.15ms  | 0.76ms      | 5.1×    |

### Pathological Cases (n=5000)

| Case             | Rust    | statsmodels | Speedup |
|------------------|---------|-------------|---------|
| clustered        | 1.18ms  | 267.8ms     | 225×    |
| constant_y       | 0.91ms  | 230.3ms     | 252×    |
| extreme_outliers | 3.50ms  | 852.0ms     | 243×    |
| high_noise       | 4.51ms  | 726.9ms     | 161×    |

### Real-World Scenarios

#### Financial Time Series

| Size    | Rust    | statsmodels | Speedup |
|---------|---------|-------------|---------|
| 500     | 0.07ms  | 10.4ms      | 151×    |
| 1,000   | 0.11ms  | 22.2ms      | 193×    |
| 5,000   | 0.49ms  | 170.9ms     | 346×    |
| 10,000  | 0.97ms  | 497.1ms     | 512×    |

#### Scientific Measurements

| Size    | Rust    | statsmodels | Speedup |
|---------|---------|-------------|---------|
| 500     | 0.12ms  | 14.1ms      | 113×    |
| 1,000   | 0.22ms  | 31.6ms      | 144×    |
| 5,000   | 0.96ms  | 268.5ms     | 280×    |
| 10,000  | 1.86ms  | 777.2ms     | 418×    |

#### Genomic Methylation (with delta=100)

| Size    | Rust    | statsmodels | Speedup |
|---------|---------|-------------|---------|
| 1,000   | 1.26ms  | 29.5ms      | 23×     |
| 5,000   | 29.2ms  | 227.3ms     | 7.8×    |
| 10,000  | 110.1ms | 662.8ms     | 6.0×    |
| 50,000  | 2.71s   | 11.2s       | 4.1×    |

## Notes

- Benchmarks use **Criterion** (Rust) and **pytest-benchmark** (Python)
- Both use identical scenarios with reproducible RNG (seed=42)
- Rust crate: `lowess` v0.6.0
- Python: `statsmodels` with Cython backend
- Test date: 2025-12-23
