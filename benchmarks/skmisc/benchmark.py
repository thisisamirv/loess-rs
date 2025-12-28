"""
Industry-level LOWESS benchmarks with JSON output for comparison with Rust.

Benchmarks are aligned with the Rust criterion benchmarks to enable direct comparison.
Results are written to benchmarks/output/skmisc_benchmark.json.

Run with: python3 benchmark.py
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Use skmisc.loess instead of statsmodels
try:
    from skmisc.loess import loess
except ImportError:
    raise ImportError("skmisc is required for these benchmarks. Please install scikit-misc.")


# ============================================================================
# Benchmark Result Storage
# ============================================================================


@dataclass
class BenchmarkResult:
    """Store benchmark timing results."""
    name: str
    size: int
    iterations: int
    mean_time_ms: float = 0.0
    std_time_ms: float = 0.0
    median_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0


def run_benchmark(name: str, size: int, func, iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
    """Run a benchmark with warmup and timing."""
    result = BenchmarkResult(name=name, size=size, iterations=iterations)
    
    # Warmup runs
    for _ in range(warmup):
        try:
            func()
        except Exception as e:
            print(f"Benchmark {name} failed during warmup: {e}")
            return result
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            func()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        except Exception as e:
            print(f"Benchmark {name} failed: {e}")
            return result
    
    if times:
        result.mean_time_ms = np.mean(times)
        result.std_time_ms = np.std(times)
        result.median_time_ms = np.median(times)
        result.min_time_ms = np.min(times)
        result.max_time_ms = np.max(times)
    
    return result


# ============================================================================
# Data Generation (Aligned with Rust)
# ============================================================================


def generate_sine_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate smooth sinusoidal data with Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, size)
    y = np.sin(x) + rng.normal(0, 0.2, size)
    return x, y


def generate_outlier_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data with 5% outliers."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, size)
    y = np.sin(x) + rng.normal(0, 0.2, size)
    
    # Add 5% outliers
    n_outliers = size // 20
    indices = rng.choice(size, n_outliers, replace=False)
    y[indices] += rng.uniform(-5, 5, n_outliers)
    return x, y


def generate_financial_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate financial time series (geometric Brownian motion)."""
    rng = np.random.default_rng(seed)
    x = np.arange(size, dtype=float)
    
    # Simulate stock prices with realistic volatility
    y = np.zeros(size)
    y[0] = 100.0
    returns = rng.normal(0.0005, 0.02, size - 1)
    for i in range(1, size):
        y[i] = y[i-1] * (1 + returns[i-1])
    return x, y


def generate_scientific_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate scientific measurement data (exponential decay with oscillations)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, size)
    signal = np.exp(-x * 0.3) * np.cos(x * 2 * np.pi)
    noise = rng.normal(0, 0.05, size)
    return x, signal + noise


def generate_genomic_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate genomic methylation data (beta values 0-1)."""
    rng = np.random.default_rng(seed)
    x = np.arange(size) * 1000.0  # CpG positions
    base = 0.5 + np.sin(x / 50000.0) * 0.3
    noise = rng.normal(0, 0.1, size)
    y = np.clip(base + noise, 0.0, 1.0)
    return x, y


def generate_clustered_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clustered x-values (groups with tiny spacing)."""
    rng = np.random.default_rng(seed)
    # i // 100 creates groups, i % 100 * 1e-6 adds tiny jitter
    x = np.array([(i // 100) + (i % 100) * 1e-6 for i in range(size)])
    y = np.sin(x) + rng.normal(0, 0.1, size)
    return x, y


def generate_high_noise_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate high-noise data (SNR < 1)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, size)
    signal = np.sin(x) * 0.5
    noise = rng.normal(0, 2.0, size)  # High noise
    return x, signal + noise


def generate_2d_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D surface data (sombrero function)."""
    rng = np.random.default_rng(seed)
    
    # Grid side length
    side = int(np.sqrt(size))
    actual_size = side * side
    
    # Create grid
    u = np.linspace(-2.0, 2.0, side)
    v = np.linspace(-2.0, 2.0, side)
    U, V = np.meshgrid(u, v)
    
    X = U.flatten()
    Y = V.flatten()
    # Combine into (N, 2) array
    predictors = np.column_stack((X, Y))
    
    # Add small jitter to predictors to avoid singularities on exact grid
    predictors += rng.normal(0, 1e-5, predictors.shape)
    
    # Sombrero function
    R = np.sqrt(predictors[:, 0]**2 + predictors[:, 1]**2)
    Z = np.where(R == 0, 1.0, np.sin(R * np.pi) / (R * np.pi))
    
    noise = rng.normal(0, 0.1, actual_size)
    response = Z + noise
    
    return predictors, response


def generate_3d_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 3D volume data."""
    rng = np.random.default_rng(seed)
    
    # Cube side length
    side = int(np.cbrt(size))
    actual_size = side * side * side
    
    u = np.linspace(0, 1, side)
    v = np.linspace(0, 1, side)
    w = np.linspace(0, 1, side)
    U, V, W = np.meshgrid(u, v, w)
    
    X = U.flatten()
    Y = V.flatten()
    Z = W.flatten()
    
    predictors = np.column_stack((X, Y, Z))
    
    # Add small jitter to predictors
    predictors += rng.normal(0, 1e-5, predictors.shape)
    
    val = np.sqrt(predictors[:, 0]**2 + predictors[:, 1]**2 + predictors[:, 2]**2)
    noise = rng.normal(0, 0.1, actual_size)
    response = val + noise
    
    return predictors, response


# ============================================================================
# Benchmark Categories (Aligned with Rust criterion benchmarks)
# ============================================================================


def benchmark_scalability(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark performance scaling with dataset size."""
    print("\n" + "=" * 80)
    print("SCALABILITY")
    print("=" * 80)
    
    results = []
    sizes = [1000, 5000]
    
    for size in sizes:
        x, y = generate_sine_data(size, seed=42)
        
        def run():
            # span=0.1, degree=1 (linear)
            l = loess(x, y, span=0.1, degree=1)
            l.fit()
        
        result = run_benchmark(f"scale_{size}", size, run, iterations)
        results.append(result)
        print(f"  scale_{size}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_fraction(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark different smoothing fractions."""
    print("\n" + "=" * 80)
    print("FRACTION")
    print("=" * 80)
    
    results = []
    size = 5000
    fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.67]
    x, y = generate_sine_data(size, seed=42)
    
    for frac in fractions:
        def run(f=frac):
            l = loess(x, y, span=f, degree=1)
            l.fit()
        
        result = run_benchmark(f"fraction_{frac}", size, run, iterations)
        results.append(result)
        print(f"  fraction_{frac}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_iterations(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark different robustness iterations."""
    print("\n" + "=" * 80)
    print("ITERATIONS")
    print("=" * 80)
    
    results = []
    size = 5000
    # skmisc controls iterations via 'control' parameter
    iter_values = [0, 1, 2, 3, 5, 10]
    x, y = generate_outlier_data(size, seed=42)
    
    for it in iter_values:
        def run(n=it):
            family = "gaussian" if n == 0 else "symmetric"
            # Pass iterations in control dict if supported, otherwise just rely on family default (usually 4)
            # skmisc loess allows passing control object or dict?
            # From docs, control is a structure. Usually we pass kwargs to loess constructor that go to control.
            # 'iterations' is the parameter for robustness steps.
            l = loess(x, y, span=0.2, degree=1, family=family, iterations=n)
            l.fit()
        
        result = run_benchmark(f"iterations_{it}", size, run, iterations)
        results.append(result)
        print(f"  iterations_{it}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results




def benchmark_financial(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark with financial time series data."""
    print("\n" + "=" * 80)
    print("FINANCIAL")
    print("=" * 80)
    
    results = []
    sizes = [500, 1000, 5000]
    
    for size in sizes:
        x, y = generate_financial_data(size, seed=42)
        
        def run():
            l = loess(x, y, span=0.1, degree=1)
            l.fit()
        
        result = run_benchmark(f"financial_{size}", size, run, iterations)
        results.append(result)
        print(f"  financial_{size}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_scientific(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark with scientific measurement data."""
    print("\n" + "=" * 80)
    print("SCIENTIFIC")
    print("=" * 80)
    
    results = []
    sizes = [500, 1000, 5000]
    
    for size in sizes:
        x, y = generate_scientific_data(size, seed=42)
        
        def run():
            l = loess(x, y, span=0.15, degree=1)
            l.fit()
        
        result = run_benchmark(f"scientific_{size}", size, run, iterations)
        results.append(result)
        print(f"  scientific_{size}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_genomic(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark with genomic methylation data."""
    print("\n" + "=" * 80)
    print("GENOMIC")
    print("=" * 80)
    
    results = []
    sizes = [1000, 5000]
    
    for size in sizes:
        x, y = generate_genomic_data(size, seed=42)
        
        def run():
            l = loess(x, y, span=0.1, degree=1)
            l.fit()
        
        result = run_benchmark(f"genomic_{size}", size, run, iterations)
        results.append(result)
        print(f"  genomic_{size}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_pathological(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark with pathological/edge case data."""
    print("\n" + "=" * 80)
    print("PATHOLOGICAL")
    print("=" * 80)
    
    results = []
    size = 5000
    
    # Clustered
    x_clustered, y_clustered = generate_clustered_data(size, seed=42)
    def run_clustered():
        l = loess(x_clustered, y_clustered, span=0.3, degree=1)
        l.fit()
    results.append(run_benchmark("clustered", size, run_clustered, iterations))
    print(f"  clustered: {results[-1].mean_time_ms:.2f} ms")
    
    # High noise
    x_noisy, y_noisy = generate_high_noise_data(size, seed=42)
    def run_noise():
        l = loess(x_noisy, y_noisy, span=0.5, degree=1, family="symmetric")
        l.fit()
    results.append(run_benchmark("high_noise", size, run_noise, iterations))
    print(f"  high_noise: {results[-1].mean_time_ms:.2f} ms")
    
    # Extreme outliers
    x_outlier, y_outlier = generate_outlier_data(size, seed=42)
    def run_outliers():
        l = loess(x_outlier, y_outlier, span=0.2, degree=1, family="symmetric", iterations=10)
        l.fit()
    results.append(run_benchmark("extreme_outliers", size, run_outliers, iterations))
    print(f"  extreme_outliers: {results[-1].mean_time_ms:.2f} ms")
    
    # Constant y
    x_const = np.arange(size, dtype=float)
    y_const = np.full(size, 5.0)
    def run_const():
        l = loess(x_const, y_const, span=0.2, degree=1)
        l.fit()
    results.append(run_benchmark("constant_y", size, run_const, iterations))
    print(f"  constant_y: {results[-1].mean_time_ms:.2f} ms")
    
    return results


def benchmark_polynomial_degrees(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark different polynomial degrees."""
    print("\n" + "=" * 80)
    print("POLYNOMIAL DEGREES")
    print("=" * 80)
    
    results = []
    size = 5000
    x, y = generate_sine_data(size, seed=42)
    
    # skmisc supports 0, 1, 2
    degrees = [
        ("linear", 1),
        ("quadratic", 2),
        ("constant", 0),
    ]
    
    for name, deg in degrees:
        def run(d=deg):
            l = loess(x, y, span=0.2, degree=d)
            l.fit()
        
        result = run_benchmark(f"degree_{name}", size, run, iterations)
        results.append(result)
        print(f"  degree_{name}: {result.mean_time_ms:.2f} ms ± {result.std_time_ms:.2f} ms")
    
    return results


def benchmark_dimensions(iterations: int = 10) -> List[BenchmarkResult]:
    """Benchmark different dimensions."""
    print("\n" + "=" * 80)
    print("DIMENSIONS")
    print("=" * 80)
    
    results = []
    
    # 1D
    size_1d = 2000
    x1, y1 = generate_sine_data(size_1d, seed=42)
    def run_1d():
        l = loess(x1, y1, span=0.3, degree=1)
        l.fit()
    results.append(run_benchmark("1d_linear", size_1d, run_1d, iterations))
    print(f"  1d_linear: {results[-1].mean_time_ms:.2f} ms")
    
    # 2D
    # generate_2d_data returns size*size points? No, it takes total size.
    # Actually my implementation of generate_2d_data calculates side=sqrt(size) then actual_size=side*side
    # So if I pass 2000, I get ~1936 points.
    x2, y2 = generate_2d_data(2000, seed=42)
    size_2d = len(y2)
    def run_2d():
        l = loess(x2, y2, span=0.3, degree=1)
        l.fit()
    results.append(run_benchmark("2d_linear", size_2d, run_2d, iterations))
    print(f"  2d_linear: {results[-1].mean_time_ms:.2f} ms")
    
    # 3D
    x3, y3 = generate_3d_data(2000, seed=42)
    size_3d = len(y3)
    def run_3d():
        l = loess(x3, y3, span=0.3, degree=1)
        l.fit()
    results.append(run_benchmark("3d_linear", size_3d, run_3d, iterations))
    print(f"  3d_linear: {results[-1].mean_time_ms:.2f} ms")
    
    return results


def main():
    """Run all benchmarks and save results."""
    print("=" * 80)
    print("SKMISC LOESS BENCHMARK SUITE (Aligned with Rust)")
    print("=" * 80)
    
    iterations = 10
    all_results: Dict[str, List[BenchmarkResult]] = {}
    
    # Run all benchmark categories
    all_results["scalability"] = benchmark_scalability(iterations)
    all_results["fraction"] = benchmark_fraction(iterations)
    all_results["iterations"] = benchmark_iterations(iterations)
    all_results["financial"] = benchmark_financial(iterations)
    all_results["scientific"] = benchmark_scientific(iterations)
    all_results["genomic"] = benchmark_genomic(iterations)
    all_results["pathological"] = benchmark_pathological(iterations)
    all_results["polynomial_degrees"] = benchmark_polynomial_degrees(iterations)
    all_results["dimensions"] = benchmark_dimensions(iterations)
    
    # Convert to JSON-serializable format
    output = {}
    for category, results in all_results.items():
        output[category] = [asdict(r) for r in results]
    
    # Save to output directory
    script_dir = Path(__file__).resolve().parent
    benchmarks_dir = script_dir.parent
    out_dir = benchmarks_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "skmisc_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Results saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
