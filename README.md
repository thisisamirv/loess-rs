# Loess Validation & Benchmarking Workspace

This workspace is dedicated to validating the correctness and benchmarking the performance of the [loess-rs](https://github.com/thisisamirv/loess-rs) Rust crate against the reference R implementation (`loess`).

It builds the `loess-rs` crate from the `develop` branch (git dependency) to ensure the latest changes are tested.

## structure

- `benchmarks/`: Performance benchmarking suite.
- `validation/`: Correctness validation suite.

## How to Run Benchmarks

Benchmarks measure execution time across various scenarios (scalability, fractions, robustness iterations, real-world data, pathological cases).

### 1. Run Rust Benchmarks (Criterion)

```bash
cd benchmarks/loess-rs
cargo bench
```

*Results are stored in `benchmarks/loess-rs/target/criterion/` with HTML reports.*

### 2. Convert Criterion Results to JSON

```bash
cd benchmarks
python3 convert_criterion.py
```

*Output: `benchmarks/output/rust_benchmark.json`*

### 3. Run R Benchmarks

```bash
# from benchmarks directory
Rscript R/benchmark.R
```

*Output: `benchmarks/output/r_benchmark.json`*

### 4. Compare Benchmark Results

Generate a comparison report showing speedups and regressions.

```bash
cd benchmarks
python3 compare_benchmark.py
```

*See `benchmarks/INTERPRETATION.md` for analysis.*

## How to Run Validation

Validation ensures the Rust implementation produces results identical (or acceptable close) to `statsmodels`.

### 1. Run Rust Validation

```bash
cd validation/loess-rs
cargo run --bin validate
```

*Output: `validation/output/rust_validate.json`*

### 2. Run R Validation

```bash
# from the validation directory
cd validation
Rscript R/validate.R
```

*Output: `validation/output/r/*.json`*

### 3. Compare Validation Results

Check for mismatches in smoothed values, residuals, and diagnostics.

```bash
cd validation
python3 compare_validation.py
```

*See `validation/INTERPRETATION.md` for analysis.*

## Requirements

- **Rust**: Latest stable.
- **R**: Latest stable with standard packages (`stats`, `jsonlite`).
- **Python**: 3.x with `numpy`, `scipy` (for comparison scripts).
