#' Industry-level LOWESS benchmarks for R with JSON output for comparison.
#'
#' Benchmarks are aligned with the Rust criterion benchmarks to enable direct comparison.
#' Results are written to benchmarks/output/r_benchmark.json.
#'
#' Run with: Rscript benchmark.R

library(jsonlite)
library(stats)

# ============================================================================
# Benchmark Result Storage
# ============================================================================

run_benchmark <- function(name, size, func, iterations = 10, warmup = 2) {
  cat(sprintf("Running benchmark: %s (size: %d)\n", name, size))

  # Warmup runs
  for (i in seq_len(warmup)) {
    tryCatch(
      {
        func()
      },
      error = function(e) {
        cat(sprintf("Benchmark %s failed during warmup: %s\n", name, e$message))
      }
    )
  }

  # Timed runs
  times <- numeric(iterations)
  for (i in seq_len(iterations)) {
    start <- Sys.time()
    tryCatch(
      {
        func()
        end <- Sys.time()
        elapsed <- as.numeric(difftime(end, start, units = "secs"))
        times[i] <- elapsed * 1000 # convert to ms
      },
      error = function(e) {
        cat(sprintf("Benchmark %s failed: %s\n", name, e$message))
      }
    )
  }

  list(
    name = name,
    size = size,
    iterations = iterations,
    mean_time_ms = mean(times),
    std_time_ms = sd(times),
    median_time_ms = median(times),
    min_time_ms = min(times),
    max_time_ms = max(times)
  )
}

# ============================================================================
# Data Generation (Aligned with Rust/Python)
# ============================================================================

generate_sine_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10, length.out = size)
  y <- sin(x) + rnorm(size, mean = 0, sd = 0.2)
  list(x = x, y = y)
}

generate_outlier_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10, length.out = size)
  y <- sin(x) + rnorm(size, mean = 0, sd = 0.2)

  n_outliers <- floor(size / 20)
  indices <- sample(seq_len(size), n_outliers)
  y[indices] <- y[indices] + runif(n_outliers, -5, 5)
  list(x = x, y = y)
}

generate_financial_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, size - 1)
  y <- numeric(size)
  y[1] <- 100.0
  returns <- rnorm(size - 1, mean = 0.0005, sd = 0.02)
  for (i in 2:size) {
    y[i] <- y[i - 1] * (1 + returns[i - 1])
  }
  list(x = x, y = y)
}

generate_scientific_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10, length.out = size)
  signal <- exp(-x * 0.3) * cos(x * 2 * pi)
  noise <- rnorm(size, mean = 0, sd = 0.05)
  list(x = x, y = signal + noise)
}

generate_genomic_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, size - 1) * 1000.0
  base <- 0.5 + sin(x / 50000.0) * 0.3
  noise <- rnorm(size, mean = 0, sd = 0.1)
  y <- pmin(pmax(base + noise, 0.0), 1.0)
  list(x = x, y = y)
}

generate_clustered_data <- function(size, seed = 42) {
  set.seed(seed)
  i <- seq(0, size - 1)
  x <- (i %/% 100) + (i %% 100) * 1e-6
  y <- sin(x) + rnorm(size, mean = 0, sd = 0.1)
  list(x = x, y = y)
}

generate_high_noise_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10, length.out = size)
  signal <- sin(x) * 0.5
  noise <- rnorm(size, mean = 0, sd = 2.0)
  list(x = x, y = signal + noise)
}

generate_2d_data <- function(size, seed = 42) {
  set.seed(seed)
  side <- floor(sqrt(size))
  actual_size <- side * side

  u <- seq(-2.0, 2.0, length.out = side)
  v <- seq(-2.0, 2.0, length.out = side)
  grid <- expand.grid(u, v)

  predictors <- as.matrix(grid)
  # Add small jitter
  predictors <- predictors + matrix(rnorm(actual_size * 2, mean = 0, sd = 1e-5), ncol = 2)

  r <- sqrt(predictors[, 1]^2 + predictors[, 2]^2)
  z <- ifelse(r == 0, 1.0, sin(r * pi) / (r * pi))

  noise <- rnorm(actual_size, mean = 0, sd = 0.1)
  list(x = predictors, y = z + noise)
}

generate_3d_data <- function(size, seed = 42) {
  set.seed(seed)
  side <- floor(size^(1 / 3))
  actual_size <- side * side * side

  u <- seq(0, 1, length.out = side)
  v <- seq(0, 1, length.out = side)
  w <- seq(0, 1, length.out = side)
  grid <- expand.grid(u, v, w)

  predictors <- as.matrix(grid)
  # Add small jitter
  predictors <- predictors + matrix(rnorm(actual_size * 3, mean = 0, sd = 1e-5), ncol = 3)

  val <- sqrt(rowSums(predictors^2))
  noise <- rnorm(actual_size, mean = 0, sd = 0.1)
  list(x = predictors, y = val + noise)
}

# ============================================================================
# Benchmark Categories
# ============================================================================

benchmark_scalability <- function(iterations = 10) {
  results <- list()
  sizes <- c(1000, 5000)

  for (size in sizes) {
    data <- generate_sine_data(size)
    run <- function() {
      loess(y ~ x, data = data, span = 0.1, degree = 1, control = loess.control(surface = "interpolate"))
    }
    results[[paste0("scale_", size)]] <- run_benchmark(paste0("scale_", size), size, run, iterations)
  }
  results
}

benchmark_fraction <- function(iterations = 10) {
  results <- list()
  size <- 5000
  fractions <- c(0.05, 0.1, 0.2, 0.3, 0.5, 0.67)
  data <- generate_sine_data(size)

  for (frac in fractions) {
    run <- function() {
      loess(y ~ x, data = data, span = frac, degree = 1, control = loess.control(surface = "interpolate"))
    }
    results[[paste0("fraction_", frac)]] <- run_benchmark(paste0("fraction_", frac), size, run, iterations)
  }
  results
}

benchmark_iterations <- function(iterations = 10) {
  results <- list()
  size <- 5000
  iter_values <- c(0, 1, 2, 3, 5, 10)
  data <- generate_outlier_data(size)

  for (it in iter_values) {
    run <- function() {
      family <- if (it == 0) "gaussian" else "symmetric"
      loess(y ~ x,
        data = data, span = 0.2, degree = 1, family = family,
        control = loess.control(iterations = it + 1, surface = "interpolate")
      )
    }
    results[[paste0("iterations_", it)]] <- run_benchmark(paste0("iterations_", it), size, run, iterations)
  }
  results
}

benchmark_financial <- function(iterations = 10) {
  results <- list()
  sizes <- c(500, 1000, 5000)

  for (size in sizes) {
    data <- generate_financial_data(size)
    run <- function() {
      loess(y ~ x, data = data, span = 0.1, degree = 1, control = loess.control(surface = "interpolate"))
    }
    results[[paste0("financial_", size)]] <- run_benchmark(paste0("financial_", size), size, run, iterations)
  }
  results
}

benchmark_scientific <- function(iterations = 10) {
  results <- list()
  sizes <- c(500, 1000, 5000)

  for (size in sizes) {
    data <- generate_scientific_data(size)
    run <- function() {
      loess(y ~ x, data = data, span = 0.15, degree = 1, control = loess.control(surface = "interpolate"))
    }
    results[[paste0("scientific_", size)]] <- run_benchmark(paste0("scientific_", size), size, run, iterations)
  }
  results
}

benchmark_genomic <- function(iterations = 10) {
  results <- list()
  sizes <- c(1000, 5000)

  for (size in sizes) {
    data <- generate_genomic_data(size)
    run <- function() {
      loess(y ~ x, data = data, span = 0.1, degree = 1, control = loess.control(surface = "interpolate"))
    }
    results[[paste0("genomic_", size)]] <- run_benchmark(paste0("genomic_", size), size, run, iterations)
  }
  results
}

benchmark_pathological <- function(iterations = 10) {
  results <- list()
  size <- 5000

  # Clustered
  data_clustered <- generate_clustered_data(size)
  run_clustered <- function() {
    loess(y ~ x, data = data_clustered, span = 0.3, degree = 1, control = loess.control(surface = "interpolate"))
  }
  results$clustered <- run_benchmark("clustered", size, run_clustered, iterations)

  # High noise
  data_noisy <- generate_high_noise_data(size)
  run_noise <- function() {
    loess(y ~ x, data = data_noisy, span = 0.5, degree = 1, family = "symmetric", control = loess.control(surface = "interpolate"))
  }
  results$high_noise <- run_benchmark("high_noise", size, run_noise, iterations)

  # Extreme outliers
  data_outlier <- generate_outlier_data(size)
  run_outliers <- function() {
    loess(y ~ x,
      data = data_outlier, span = 0.2, degree = 1, family = "symmetric",
      control = loess.control(iterations = 11, surface = "interpolate")
    )
  }
  results$extreme_outliers <- run_benchmark("extreme_outliers", size, run_outliers, iterations)

  # Constant y
  x_const <- seq(size)
  y_const <- rep(5.0, size)
  data_const <- list(x = x_const, y = y_const)
  run_const <- function() {
    loess(y ~ x, data = data_const, span = 0.2, degree = 1, control = loess.control(surface = "interpolate"))
  }
  results$constant_y <- run_benchmark("constant_y", size, run_const, iterations)

  results
}

benchmark_polynomial_degrees <- function(iterations = 10) {
  results <- list()
  size <- 5000
  data <- generate_sine_data(size)

  degrees <- list(
    list(name = "linear", deg = 1),
    list(name = "quadratic", deg = 2)
    # R loess doesn't support degree 0 in the same way, usually it's degree 1 or 2.
  )

  for (d in degrees) {
    run <- function() {
      loess(y ~ x, data = data, span = 0.2, degree = d$deg, control = loess.control(surface = "interpolate"))
    }
    results[[paste0("degree_", d$name)]] <- run_benchmark(paste0("degree_", d$name), size, run, iterations)
  }
  results
}

benchmark_dimensions <- function(iterations = 10) {
  results <- list()

  # 1D
  size_1d <- 2000
  data_1d <- generate_sine_data(size_1d)
  run_1d <- function() {
    loess(y ~ x, data = data_1d, span = 0.3, degree = 1, control = loess.control(surface = "interpolate"))
  }
  results$linear_1d <- run_benchmark("1d_linear", size_1d, run_1d, iterations)

  # 2D
  data_2d <- generate_2d_data(2000)
  size_2d <- length(data_2d$y)
  run_2d <- function() {
    loess(y ~ x, data = data_2d, span = 0.3, degree = 1, control = loess.control(surface = "interpolate"))
  }
  results$linear_2d <- run_benchmark("2d_linear", size_2d, run_2d, iterations)

  # 3D
  data_3d <- generate_3d_data(2000)
  size_3d <- length(data_3d$y)
  run_3d <- function() {
    loess(y ~ x, data = data_3d, span = 0.3, degree = 1, control = loess.control(surface = "interpolate"))
  }
  results$linear_3d <- run_benchmark("3d_linear", size_3d, run_3d, iterations)

  results
}

# ============================================================================
# Main Execution
# ============================================================================

main <- function() {
  cat("============================================================================\n")
  cat("R LOESS BENCHMARK SUITE (Aligned with Python/Rust)\n")
  cat("============================================================================\n")

  iterations <- 10
  all_results <- list()

  all_results$scalability <- unname(benchmark_scalability(iterations))
  all_results$fraction <- unname(benchmark_fraction(iterations))
  all_results$iterations <- unname(benchmark_iterations(iterations))
  all_results$financial <- unname(benchmark_financial(iterations))
  all_results$scientific <- unname(benchmark_scientific(iterations))
  all_results$genomic <- unname(benchmark_genomic(iterations))
  all_results$pathological <- unname(benchmark_pathological(iterations))
  all_results$polynomial_degrees <- unname(benchmark_polynomial_degrees(iterations))
  all_results$dimensions <- unname(benchmark_dimensions(iterations))

  # Move to output directory
  out_dir <- "output"
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }

  out_path <- file.path(out_dir, "r_benchmark.json")
  write_json(all_results, out_path, auto_unbox = TRUE, pretty = TRUE)

  cat("\n============================================================================\n")
  cat(sprintf("Results saved to %s\n", out_path))
  cat("============================================================================\n")
}

if (interactive() == FALSE) {
  main()
}
