//! Execution engine for LOESS smoothing operations.
//!
//! ## Purpose
//!
//! This module provides the core execution engine that orchestrates LOESS
//! smoothing operations. It handles the iteration loop, robustness weight
//! updates, convergence checking, cross-validation, and variance estimation.
//! The executor is the central component that coordinates all lower-level
//! algorithms to produce smoothed results.
//!
//! ## Design notes
//!
//! * Provides both configuration-based and parameter-based entry points.
//! * Handles cross-validation for automatic fraction selection.
//! * Supports auto-convergence for adaptive iteration counts.
//! * Manages working buffers efficiently to minimize allocations.
//! * Uses delta optimization for performance on dense data.
//! * Separates concerns: fitting, interpolation, robustness, convergence.
//! * Generic over `Float` types to support f32 and f64.
//!
//! ## Key concepts
//!
//! * **Execution Loop**: The central iteration cycle (Fit -> Residuals -> Weights -> Repeat).
//! * **Auto-convergence**: Dynamically stopping iterations when parameters stabilize.
//! * **Delta Optimization**: Skipping expensive re-fitting for nearby points.
//!
//! ## Invariants
//!
//! * Input x-values are assumed to be monotonically increasing (sorted).
//! * All working buffers have the same length as input data.
//! * Robustness weights are always in [0, 1].
//! * Window size is at least 2 and at most n.
//! * Iteration count is non-negative.
//!
//! ## Non-goals
//!
//! * This module does not validate input data (handled by `validator`).
//! * This module does not sort input data (caller's responsibility).
//! * This module does not provide public-facing result formatting.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering::Equal;
use core::fmt::Debug;
use num_traits::Float;

// Internal dependencies
use crate::algorithms::interpolation::InterpolationSurface;
use crate::algorithms::regression::{
    FittingBuffer, PolynomialDegree, RegressionContext, ZeroWeightFallback,
};
use crate::algorithms::robustness::RobustnessMethod;
use crate::evaluation::cv::CVKind;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::BoundaryPolicy;
use crate::math::distance::DistanceMetric;
use crate::math::kernel::WeightFunction;
use crate::math::neighborhood::{KDTree, Neighborhood, PointDistance, floyd_rivest_select};
use crate::primitives::backend::Backend;
use crate::primitives::window::Window;

/// Standard LOESS distance calculator.
///
/// Implements `PointDistance` using either Euclidean or Normalized Euclidean metrics.
pub struct LoessDistanceCalculator<'a, T: Float> {
    /// The distance metric to use (Euclidean or Normalized).
    pub metric: DistanceMetric<T>,
    /// Normalization scales for each dimension (used if metric is Normalized).
    pub scales: &'a [T],
}

impl<'a, T: Float> PointDistance<T> for LoessDistanceCalculator<'a, T> {
    fn distance(&self, a: &[T], b: &[T]) -> T {
        match &self.metric {
            DistanceMetric::Normalized => DistanceMetric::normalized(a, b, self.scales),
            DistanceMetric::Euclidean => DistanceMetric::euclidean(a, b),
            DistanceMetric::Manhattan => DistanceMetric::manhattan(a, b),
            DistanceMetric::Chebyshev => DistanceMetric::chebyshev(a, b),
            DistanceMetric::Minkowski(p) => DistanceMetric::minkowski(a, b, *p),
            DistanceMetric::Weighted(w) => DistanceMetric::weighted(a, b, w),
        }
    }

    fn split_distance(&self, dim: usize, split_val: T, query_val: T) -> T {
        let diff = (query_val - split_val).abs();
        match &self.metric {
            DistanceMetric::Normalized => diff * self.scales[dim],
            DistanceMetric::Euclidean => diff,
            DistanceMetric::Manhattan => diff,
            DistanceMetric::Chebyshev => diff,
            DistanceMetric::Minkowski(_) => diff,
            DistanceMetric::Weighted(w) => diff * w[dim].sqrt(),
        }
    }
}

// ============================================================================
// Surface Mode
// ============================================================================

/// Mode for surface evaluation.
///
/// Controls whether to use interpolation surface (faster, less accurate) or
/// direct per-point fitting (slower, more accurate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceMode {
    /// Use interpolation surface for faster evaluation.
    #[default]
    Interpolation,

    /// Use direct per-point fitting for maximum accuracy.
    Direct,
}

// ============================================================================
// Type Definitions
// ============================================================================

/// Signature for custom smooth pass function
#[doc(hidden)]
pub type SmoothPassFn<T> = fn(
    &[T],               // x
    &[T],               // y
    usize,              // window_size
    T,                  // delta (interpolation optimization threshold)
    bool,               // use_robustness
    &[T],               // robustness_weights
    &mut [T],           // output (y_smooth)
    WeightFunction,     // weight_function
    ZeroWeightFallback, // zero_weight_flag
);

/// Signature for custom cross-validation pass function
#[doc(hidden)]
pub type CVPassFn<T> = fn(
    &[T],            // x
    &[T],            // y
    &[T],            // candidate fractions
    CVKind,          // CV strategy
    &LoessConfig<T>, // Config for internal fits
) -> (T, Vec<T>); // (best_fraction, scores)

/// Signature for custom interval estimation pass function
#[doc(hidden)]
pub type IntervalPassFn<T> = fn(
    &[T],               // x
    &[T],               // y
    &[T],               // y_smooth
    usize,              // window_size
    &[T],               // robustness_weights
    WeightFunction,     // weight_function
    &IntervalMethod<T>, // interval configuration
) -> Vec<T>; // standard errors

/// Signature for custom iteration batch pass function (GPU acceleration).
#[doc(hidden)]
pub type FitPassFn<T> = fn(
    &[T],            // x
    &[T],            // y
    &LoessConfig<T>, // full configuration
) -> (
    Vec<T>,         // smoothed
    Option<Vec<T>>, // std_errors
    usize,          // iterations
    Vec<T>,         // robustness_weights
);

/// Output from LOESS execution.
#[derive(Debug, Clone)]
pub struct ExecutorOutput<T> {
    /// Smoothed y-values.
    pub smoothed: Vec<T>,

    /// Standard errors (if SE estimation or intervals were requested).
    pub std_errors: Option<Vec<T>>,

    /// Number of iterations performed (if auto-convergence was active).
    pub iterations: Option<usize>,

    /// Smoothing fraction used (selected by CV or configured).
    pub used_fraction: T,

    /// RMSE scores for each tested fraction (if CV was performed).
    pub cv_scores: Option<Vec<T>>,

    /// Final robustness weights from iterative refinement.
    pub robustness_weights: Vec<T>,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for LOESS execution.
#[derive(Debug, Clone)]
pub struct LoessConfig<T> {
    /// Smoothing fraction (0, 1].
    /// If `None` and `cv_fractions` are provided, bandwidth selection is performed.
    pub fraction: Option<T>,

    /// Number of robustness iterations (0 means initial fit only).
    pub iterations: usize,

    /// Kernel weight function used for local regression.
    pub weight_function: WeightFunction,

    /// Zero-weight fallback policy.
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Robustness weighting method for outlier downweighting.
    pub robustness_method: RobustnessMethod,

    /// Candidate fractions to evaluate during cross-validation.
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation strategy (e.g., K-Fold or LOOCV).
    pub cv_kind: Option<CVKind>,

    /// Seed for random number generation in cross-validation.
    pub cv_seed: Option<u64>,

    /// Convergence tolerance for early stopping of robustness iterations.
    pub auto_convergence: Option<T>,

    /// Configuration for standard errors and intervals.
    pub return_variance: Option<IntervalMethod<T>>,

    /// Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    /// Polynomial degree for local regression (0=constant, 1=linear, 2=quadratic).
    pub polynomial_degree: PolynomialDegree,

    /// Number of predictor dimensions (default: 1).
    pub dimensions: usize,

    /// Distance metric for nD neighborhood computation.
    pub distance_metric: DistanceMetric<T>,

    /// Surface evaluation mode (Interpolation or Direct).
    pub surface_mode: SurfaceMode,

    /// Maximum number of vertices for the interpolation surface.
    pub interpolation_vertices: Option<usize>,

    /// Cell size as a fraction of the smoothing span (default: 0.2).
    /// Used to determine subdivision when `surface_mode` is `Interpolation`.
    pub cell: Option<f64>,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    /// Custom smooth pass function (enables parallel execution).
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    /// Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    /// Custom iteration batch pass function for GPU acceleration.
    #[doc(hidden)]
    pub custom_fit_pass: Option<FitPassFn<T>>,

    /// Execution backend hint for extension crates.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    /// Whether to use parallel execution
    #[doc(hidden)]
    pub parallel: bool,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for LoessConfig<T> {
    fn default() -> Self {
        Self {
            fraction: None,
            iterations: 3,
            weight_function: WeightFunction::default(),
            zero_weight_fallback: ZeroWeightFallback::default(),
            robustness_method: RobustnessMethod::default(),
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            auto_convergence: None,
            return_variance: None,
            boundary_policy: BoundaryPolicy::default(),
            polynomial_degree: PolynomialDegree::default(),
            dimensions: 1,
            distance_metric: DistanceMetric::default(),
            surface_mode: SurfaceMode::default(),
            interpolation_vertices: None,
            cell: None,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_fit_pass: None,
            parallel: false,
            backend: None,
        }
    }
}

/// Unified executor for LOESS smoothing operations.
#[derive(Debug, Clone)]
pub struct LoessExecutor<T: Float> {
    /// Smoothing fraction (0, 1].
    pub fraction: T,

    /// Number of robustness iterations.
    pub iterations: usize,

    /// Kernel weight function.
    pub weight_function: WeightFunction,

    /// Zero weight fallback flag.
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Robustness method for iterative refinement.
    pub robustness_method: RobustnessMethod,

    /// Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    /// Polynomial degree for local regression.
    pub polynomial_degree: PolynomialDegree,

    /// Number of predictor dimensions.
    pub dimensions: usize,

    /// Distance metric for nD neighborhood computation.
    pub distance_metric: DistanceMetric<T>,

    /// Surface evaluation mode (Interpolation or Direct).
    pub surface_mode: SurfaceMode,

    /// Maximum number of vertices for interpolation.
    pub interpolation_vertices: Option<usize>,

    /// Cell size for interpolation subdivision.
    pub cell: Option<f64>,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    /// Custom smooth pass function (e.g., for parallel execution).
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    /// Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    /// Custom iteration batch pass function for GPU acceleration.
    #[doc(hidden)]
    pub custom_fit_pass: Option<FitPassFn<T>>,

    /// Execution backend hint for extension crates.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    /// Whether to use parallel execution
    #[doc(hidden)]
    pub parallel: bool,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for LoessExecutor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> LoessExecutor<T> {
    // ========================================================================
    // Constructor and Builder Methods
    // ========================================================================

    /// Create a new executor with default parameters.
    pub fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap()),
            iterations: 3,
            weight_function: WeightFunction::default(),
            zero_weight_fallback: ZeroWeightFallback::default(),
            robustness_method: RobustnessMethod::default(),
            boundary_policy: BoundaryPolicy::default(),
            polynomial_degree: PolynomialDegree::default(),
            dimensions: 1,
            distance_metric: DistanceMetric::default(),
            surface_mode: SurfaceMode::default(),
            interpolation_vertices: None,
            cell: None,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_fit_pass: None,
            parallel: false,
            backend: None,
        }
    }

    /// Create a new executor from a `LoessConfig`.
    pub fn from_config(config: &LoessConfig<T>) -> Self {
        let default_frac = T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap());
        Self::new()
            .fraction(config.fraction.unwrap_or(default_frac))
            .iterations(config.iterations)
            .weight_function(config.weight_function)
            .zero_weight_fallback(config.zero_weight_fallback)
            .robustness_method(config.robustness_method)
            .boundary_policy(config.boundary_policy)
            .polynomial_degree(config.polynomial_degree)
            .dimensions(config.dimensions)
            .distance_metric(config.distance_metric.clone())
            .surface_mode(config.surface_mode)
            .interpolation_vertices(config.interpolation_vertices)
            .cell(config.cell)
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            .custom_smooth_pass(config.custom_smooth_pass)
            .custom_cv_pass(config.custom_cv_pass)
            .custom_interval_pass(config.custom_interval_pass)
            .custom_fit_pass(config.custom_fit_pass)
            .parallel(config.parallel)
            .backend(config.backend)
    }

    /// Set the smoothing fraction (bandwidth).
    pub fn fraction(mut self, frac: T) -> Self {
        self.fraction = frac;
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, niter: usize) -> Self {
        self.iterations = niter;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    /// Set the zero weight fallback policy flag.
    pub fn zero_weight_fallback(mut self, flag: ZeroWeightFallback) -> Self {
        self.zero_weight_fallback = flag;
        self
    }

    /// Set the robustness method for iterative refinement.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary_policy = policy;
        self
    }

    /// Set the polynomial degree for local regression.
    pub fn polynomial_degree(mut self, degree: PolynomialDegree) -> Self {
        self.polynomial_degree = degree;
        self
    }

    /// Set the number of predictor dimensions.
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    /// Set the distance metric for nD neighborhood computation.
    pub fn distance_metric(mut self, metric: DistanceMetric<T>) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the surface evaluation mode (Interpolation or Direct).
    pub fn surface_mode(mut self, mode: SurfaceMode) -> Self {
        self.surface_mode = mode;
        self
    }

    /// Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: Option<usize>) -> Self {
        self.interpolation_vertices = vertices;
        self
    }

    /// Set the interpolation cell size.
    pub fn cell(mut self, cell: Option<f64>) -> Self {
        self.cell = cell;
        self
    }

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++

    /// Set a custom smooth pass function (e.g., for parallelization).
    #[doc(hidden)]
    pub fn custom_smooth_pass(mut self, smooth_pass_fn: Option<SmoothPassFn<T>>) -> Self {
        self.custom_smooth_pass = smooth_pass_fn;
        self
    }

    /// Set a custom cross-validation pass function.
    #[doc(hidden)]
    pub fn custom_cv_pass(mut self, cv_pass_fn: Option<CVPassFn<T>>) -> Self {
        self.custom_cv_pass = cv_pass_fn;
        self
    }

    /// Set a custom interval estimation pass function.
    #[doc(hidden)]
    pub fn custom_interval_pass(mut self, interval_pass_fn: Option<IntervalPassFn<T>>) -> Self {
        self.custom_interval_pass = interval_pass_fn;
        self
    }

    /// Set whether to use parallel execution.
    #[doc(hidden)]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the execution backend hint.
    #[doc(hidden)]
    pub fn backend(mut self, backend: Option<Backend>) -> Self {
        self.backend = backend;
        self
    }

    /// Set a custom iteration batch pass function (e.g., for GPU acceleration).
    #[doc(hidden)]
    pub fn custom_fit_pass(mut self, fit_pass_fn: Option<FitPassFn<T>>) -> Self {
        self.custom_fit_pass = fit_pass_fn;
        self
    }

    // ========================================================================
    // Main Entry Points
    // ========================================================================

    /// Smooth data using a `LoessConfig` payload.
    pub fn run_with_config(x: &[T], y: &[T], config: LoessConfig<T>) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let executor = LoessExecutor::from_config(&config);

        // Handle cross-validation if configured
        if let Some(ref cv_fracs) = config.cv_fractions {
            let cv_kind = config.cv_kind.unwrap_or(CVKind::KFold(5));
            let dims = executor.dimensions;

            // Run CV to find best fraction
            let (best_frac, scores) = if let Some(callback) = config.custom_cv_pass {
                callback(x, y, cv_fracs, cv_kind, &config)
            } else {
                let predictor = if dims > 1 {
                    Some(|train_x: &[T], train_y: &[T], test_x: &[T], f: T| {
                        let n_train = train_y.len();
                        let window_size = Window::calculate_span(n_train, f);
                        let (mins, maxs) = DistanceMetric::compute_ranges(train_x, dims);
                        let scales = DistanceMetric::compute_normalization_scales(&mins, &maxs);
                        let kdtree = KDTree::new(train_x, dims);

                        executor.predict(
                            train_x,
                            train_y,
                            &vec![T::one(); n_train],
                            test_x,
                            window_size,
                            &scales,
                            Some(&kdtree),
                        )
                    })
                } else {
                    None
                };

                cv_kind.run(
                    x,
                    y,
                    dims,
                    cv_fracs,
                    config.cv_seed,
                    |tx, ty, f| executor.run(tx, ty, Some(f), None, None, None).smoothed,
                    predictor,
                )
            };

            // Run final pass with best fraction
            let mut output = executor.run(
                x,
                y,
                Some(best_frac),
                Some(config.iterations),
                config.auto_convergence,
                config.return_variance.as_ref(),
            );
            output.cv_scores = Some(scores);
            output.used_fraction = best_frac;
            output
        } else {
            // Direct run (no CV)
            executor.run(
                x,
                y,
                config.fraction,
                Some(config.iterations),
                config.auto_convergence,
                config.return_variance.as_ref(),
            )
        }
    }

    /// Execute smoothing with explicit overrides for specific parameters.
    ///
    /// Uses interpolation surface for efficient evaluation - fits only at
    /// cell vertices and interpolates for all other points.
    ///
    /// # Special Cases
    ///
    /// * **Insufficient data** (n < 2): Returns original y-values.
    /// * **Global regression** (fraction >= 1.0): Performs OLS on the entire dataset.
    fn run(
        &self,
        x: &[T],
        y: &[T],
        fraction: Option<T>,
        max_iter: Option<usize>,
        tolerance: Option<T>,
        confidence_method: Option<&IntervalMethod<T>>,
    ) -> ExecutorOutput<T> {
        let dims = self.dimensions;
        let n = x.len() / dims;
        let eff_fraction = fraction.unwrap_or(self.fraction);

        // Calculate window size
        let window_size = Window::calculate_span(n, eff_fraction);
        let target_iterations = max_iter.unwrap_or(self.iterations);

        // Apply boundary policy (unified)
        let (ax, ay, mapping) = self.boundary_policy.apply(x, y, dims, window_size);
        let n_total = ay.len();
        let is_augmented = n_total > n;

        // Compute normalization scales using the full (augmented) range
        let mut mins = vec![T::zero(); dims];
        let mut maxs = vec![T::zero(); dims];
        // Initialize mins/maxs with first point
        mins[..dims].copy_from_slice(&ax[..dims]);
        maxs[..dims].copy_from_slice(&ax[..dims]);
        for i in 1..n_total {
            for d in 0..dims {
                let val = ax[i * dims + d];
                if val < mins[d] {
                    mins[d] = val;
                }
                if val > maxs[d] {
                    maxs[d] = val;
                }
            }
        }

        let mut scales = vec![T::one(); dims];
        for d in 0..dims {
            let range = maxs[d] - mins[d];
            if range > T::zero() {
                scales[d] = T::one() / range;
            }
        }

        // Build KD-Tree for efficient kNN at vertices
        let kdtree = KDTree::new(&ax, dims);

        // Define distance calculator
        let dist_calc = LoessDistanceCalculator {
            metric: self.distance_metric.clone(),
            scales: &scales,
        };

        // Resolution First: no default limit unless explicitly provided
        let max_vertices = self.interpolation_vertices.unwrap_or(usize::MAX);

        let mut y_smooth = vec![T::zero(); n];
        let mut robustness_weights = vec![T::one(); n_total];
        let mut residuals = vec![T::zero(); n];

        let mut iterations_performed = 0;

        // For interpolation mode: build surface once before iterations
        let mut _surface_opt: Option<InterpolationSurface<T>> = None;
        if self.surface_mode == SurfaceMode::Interpolation {
            // Initial build with unit robustness weights
            let fitter = |vertex: &[T], neighborhood: &Neighborhood<T>| {
                let mut context = RegressionContext {
                    x: &ax,
                    dimensions: dims,
                    y: &ay,
                    query_idx: 0,
                    query_point: Some(vertex),
                    neighborhood,
                    use_robustness: false,
                    robustness_weights: &robustness_weights,
                    weight_function: self.weight_function,
                    zero_weight_fallback: self.zero_weight_fallback,
                    polynomial_degree: self.polynomial_degree,
                    compute_leverage: false,
                    buffer: None,
                };
                context.fit_with_coefficients()
            };

            let surface = InterpolationSurface::build(
                &ax,
                &ay,
                dims,
                eff_fraction,
                window_size,
                &dist_calc,
                &kdtree,
                max_vertices,
                fitter,
                T::from(self.cell.unwrap_or(0.2)).unwrap_or(T::from(0.2).unwrap()),
            );

            // Evaluate surface at original data points
            for (i, val) in y_smooth.iter_mut().enumerate().take(n) {
                let query_offset = i * dims;
                let query_point = &ax[query_offset..query_offset + dims];
                *val = surface.evaluate(query_point);
            }

            _surface_opt = Some(surface);
        } else {
            // Direct mode: initial fit
            self.smooth_pass(
                &ax,
                &ay,
                window_size,
                &robustness_weights,
                false,
                &scales,
                &mut y_smooth,
                n,
                &kdtree,
            );
        }

        // Robustness iteration loop
        for iter in 1..=target_iterations {
            iterations_performed = iter;

            // Update robustness weights based on residuals from previous pass
            for i in 0..n {
                residuals[i] = (y[i] - y_smooth[i]).abs();
            }

            let mut sorted_residuals = residuals.clone();
            let median_idx = n / 2;
            floyd_rivest_select(&mut sorted_residuals, median_idx, |a, b| {
                a.partial_cmp(b).unwrap_or(Equal)
            });
            let median_residual = sorted_residuals[median_idx];
            let mad = median_residual;

            if mad <= T::epsilon() {
                break;
            }

            let tolerance_val = tolerance.unwrap_or_else(|| T::from(1e-6).unwrap());

            // 1. Compute new weights for original points
            // Use a temp buffer to avoid overwriting weights used by augmented points before sync
            let mut new_weights = vec![T::zero(); n];
            for i in 0..n {
                let r = residuals[i] / (T::from(6.0).unwrap() * mad);
                new_weights[i] = if r < T::one() {
                    let tmp = T::one() - r * r;
                    tmp * tmp
                } else {
                    T::zero()
                };
            }

            // 2. Sync to robustness_weights and check convergence
            let mut max_change = T::zero();

            if is_augmented {
                for (aug_idx, &orig_idx) in mapping.iter().enumerate() {
                    let old_w = robustness_weights[aug_idx];
                    let new_w = new_weights[orig_idx];
                    robustness_weights[aug_idx] = new_w;

                    let change = (new_w - old_w).abs();
                    if change > max_change {
                        max_change = change;
                    }
                }
            } else {
                for i in 0..n {
                    let old_w = robustness_weights[i];
                    let new_w = new_weights[i];
                    robustness_weights[i] = new_w;

                    let change = (new_w - old_w).abs();
                    if change > max_change {
                        max_change = change;
                    }
                }
            }

            if max_change < tolerance_val {
                break;
            }

            // Re-fit with new robustness weights
            match self.surface_mode {
                SurfaceMode::Interpolation => {
                    // Refit vertex values using existing surface structure
                    if let Some(ref mut surface) = _surface_opt {
                        let fitter = |vertex: &[T], neighborhood: &Neighborhood<T>| {
                            let mut context = RegressionContext {
                                x: &ax,
                                dimensions: dims,
                                y: &ay,
                                query_idx: 0,
                                query_point: Some(vertex),
                                neighborhood,
                                use_robustness: true,
                                robustness_weights: &robustness_weights,
                                weight_function: self.weight_function,
                                zero_weight_fallback: self.zero_weight_fallback,
                                polynomial_degree: self.polynomial_degree,
                                compute_leverage: false,
                                buffer: None,
                            };
                            context.fit_with_coefficients()
                        };

                        surface.refit_values(&ay, &kdtree, window_size, &dist_calc, fitter);

                        // Re-evaluate at data points
                        for (i, val) in y_smooth.iter_mut().enumerate().take(n) {
                            let query_offset = i * dims;
                            let query_point = &ax[query_offset..query_offset + dims];
                            *val = surface.evaluate(query_point);
                        }
                    }
                }
                SurfaceMode::Direct => {
                    self.smooth_pass(
                        &ax,
                        &ay,
                        window_size,
                        &robustness_weights,
                        true,
                        &scales,
                        &mut y_smooth,
                        n,
                        &kdtree,
                    );
                }
            }
        }

        // Standard errors
        let se = if confidence_method.is_some() {
            // Estimate residual standard deviation (sigma) robustly
            for i in 0..n {
                residuals[i] = (y[i] - y_smooth[i]).abs();
            }
            let mut sorted_residuals = residuals.clone();
            let median_idx = n / 2;
            floyd_rivest_select(&mut sorted_residuals, median_idx, |a, b| {
                a.partial_cmp(b).unwrap_or(Equal)
            });
            let median_residual = sorted_residuals[median_idx];
            let sigma = median_residual * T::from(1.4826).unwrap();

            // Approximate leverage based on fraction
            let approx_leverage = eff_fraction / T::from(n).unwrap();
            let se_vec: Vec<T> = (0..n).map(|_| sigma * approx_leverage.sqrt()).collect();
            Some(se_vec)
        } else {
            None
        };

        // Extract robustness weights for original points only
        let final_robustness_weights: Vec<T> = if is_augmented {
            // Find the start of the contiguous original data block [0, 1, ..., n-1]
            // within the mapping to locate the corresponding robustness weights.
            let mut start_offset = 0;
            for i in 0..n_total {
                if mapping[i] == 0 {
                    let mut match_seq = true;
                    if i + n <= n_total {
                        for j in 0..n {
                            if mapping[i + j] != j {
                                match_seq = false;
                                break;
                            }
                        }
                    } else {
                        match_seq = false;
                    }

                    if match_seq {
                        start_offset = i;
                        break;
                    }
                }
            }
            robustness_weights[start_offset..start_offset + n].to_vec()
        } else {
            robustness_weights[..n].to_vec()
        };

        ExecutorOutput {
            smoothed: y_smooth,
            std_errors: se,
            iterations: Some(iterations_performed),
            used_fraction: eff_fraction,
            cv_scores: None,
            robustness_weights: final_robustness_weights,
        }
    }

    // ========================================================================
    // Main Algorithmic Logic
    // ========================================================================

    /// Predict values at arbitrary points using the provided training data.
    ///
    /// This is used for out-of-sample prediction, specifically during cross-validation.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &self,
        x_train: &[T],
        y_train: &[T],
        robustness_weights: &[T],
        x_query: &[T],
        window_size: usize,
        scales: &[T],
        kdtree: Option<&KDTree<T>>,
    ) -> Vec<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let dims = self.dimensions;
        let n_query = x_query.len() / dims;
        let mut y_pred = vec![T::zero(); n_query];

        let dist_calc = LoessDistanceCalculator {
            metric: self.distance_metric.clone(),
            scales,
        };

        let n_coeffs = self.polynomial_degree.num_coefficients_nd(dims);
        let mut buffer = FittingBuffer::new(window_size, n_coeffs);

        for (i, pred) in y_pred.iter_mut().enumerate() {
            let query_offset = i * dims;
            let query_point = &x_query[query_offset..query_offset + dims];

            // Find neighbors in training data (KD-tree is always available)
            let neighborhood = kdtree
                .expect("KD-tree must be provided for prediction")
                .find_k_nearest(query_point, window_size, &dist_calc, None);

            // Fit local polynomial
            let mut context = RegressionContext {
                x: x_train,
                dimensions: dims,
                y: y_train,
                query_idx: 0,
                query_point: Some(query_point),
                neighborhood: &neighborhood,
                use_robustness: true,
                robustness_weights,
                weight_function: self.weight_function,
                zero_weight_fallback: ZeroWeightFallback::default(), // CV fallback
                polynomial_degree: self.polynomial_degree,
                compute_leverage: false,
                buffer: Some(&mut buffer),
            };

            if let Some((val, _)) = context.fit() {
                *pred = val;
            } else {
                // If fitting fails, fallback to something or zero
                // For CV, zero is better than crashing, but we could try to find a global mean
                *pred = T::zero();
            }
        }
        y_pred
    }

    /// Perform a single smoothing pass over all nD points (Direct mode).
    #[allow(clippy::too_many_arguments)]
    fn smooth_pass(
        &self,
        x: &[T],
        y: &[T],
        window_size: usize,
        robustness_weights: &[T],
        use_robustness: bool,
        scales: &[T],
        y_smooth: &mut [T],
        original_n: usize,
        kdtree: &KDTree<T>,
    ) where
        T: Float + Debug + Send + Sync + 'static,
    {
        let dims = self.dimensions;

        let dist_calc = LoessDistanceCalculator {
            metric: self.distance_metric.clone(),
            scales,
        };

        let n_coeffs = self.polynomial_degree.num_coefficients_nd(dims);
        let mut buffer = FittingBuffer::new(window_size, n_coeffs);

        for i in 0..original_n {
            let query_offset = i * dims;
            let query_point = &x[query_offset..query_offset + dims];
            let neighborhood = kdtree.find_k_nearest(query_point, window_size, &dist_calc, None);

            let mut context = RegressionContext {
                x,
                dimensions: dims,
                y,
                query_idx: i,
                query_point: None,
                neighborhood: &neighborhood,
                use_robustness,
                robustness_weights,
                weight_function: self.weight_function,
                zero_weight_fallback: self.zero_weight_fallback,
                polynomial_degree: self.polynomial_degree,
                compute_leverage: false,
                buffer: Some(&mut buffer),
            };

            if let Some((val, _)) = context.fit() {
                y_smooth[i] = val;
            } else {
                y_smooth[i] = y[i];
            }
        }
    }
}
