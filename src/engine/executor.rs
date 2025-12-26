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
//! * This module does not handle parallel execution directly (handled by adapters).

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use core::mem::swap;
use num_traits::Float;

// Internal dependencies
use crate::algorithms::interpolation::interpolate_gap;
use crate::algorithms::regression::{
    GLSModel, LinearRegression, Regression, RegressionContext, ZeroWeightFallback,
};
use crate::algorithms::robustness::RobustnessMethod;
use crate::evaluation::cv::CVKind;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::apply_boundary_policy;
use crate::math::kernel::WeightFunction;
use crate::primitives::backend::Backend;
use crate::primitives::partition::BoundaryPolicy;
use crate::primitives::window::Window;

// ============================================================================
// Type Definitions
// ============================================================================

/// Signature for custom smooth pass function
#[doc(hidden)]
pub type SmoothPassFn<T> = fn(
    &[T],           // x
    &[T],           // y
    usize,          // window_size
    T,              // delta (interpolation optimization threshold)
    bool,           // use_robustness
    &[T],           // robustness_weights
    &mut [T],       // output (y_smooth)
    WeightFunction, // weight_function
    u8,             // zero_weight_flag
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

/// Working buffers for LOESS iteration.
#[doc(hidden)]
pub struct IterationBuffers<T> {
    /// Current smoothed values
    pub y_smooth: Vec<T>,

    /// Previous iteration values (for convergence check)
    pub y_prev: Vec<T>,

    /// Robustness weights
    pub robustness_weights: Vec<T>,

    /// Residuals buffer
    pub residuals: Vec<T>,

    /// Kernel weights scratch buffer
    pub weights: Vec<T>,
}

impl<T: Float> IterationBuffers<T> {
    /// Allocate all working buffers for LOESS iteration.
    pub fn allocate(n: usize, use_convergence: bool) -> Self {
        Self {
            y_smooth: vec![T::zero(); n],
            y_prev: if use_convergence {
                vec![T::zero(); n]
            } else {
                Vec::new()
            },
            robustness_weights: vec![T::one(); n],
            residuals: vec![T::zero(); n],
            weights: vec![T::zero(); n],
        }
    }
}

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

    /// Delta parameter for linear interpolation optimization.
    pub delta: T,

    /// Kernel weight function used for local regression.
    pub weight_function: WeightFunction,

    /// Zero-weight fallback policy (via [`ZeroWeightFallback`]).
    pub zero_weight_fallback: u8,

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

impl<T: Float> Default for LoessConfig<T> {
    fn default() -> Self {
        Self {
            fraction: None,
            iterations: 3,
            delta: T::zero(),
            weight_function: WeightFunction::default(),
            zero_weight_fallback: 0,
            robustness_method: RobustnessMethod::default(),
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            auto_convergence: None,
            return_variance: None,
            boundary_policy: BoundaryPolicy::default(),
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

    /// Delta for interpolation optimization.
    pub delta: T,

    /// Kernel weight function.
    pub weight_function: WeightFunction,

    /// Zero weight fallback flag (0=UseLocalMean, 1=ReturnOriginal, 2=ReturnNone).
    pub zero_weight_fallback: u8,

    /// Robustness method for iterative refinement.
    pub robustness_method: RobustnessMethod,

    /// Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

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

impl<T: Float> Default for LoessExecutor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> LoessExecutor<T> {
    // ========================================================================
    // Constructor and Builder Methods
    // ========================================================================

    /// Create a new executor with default parameters.
    pub fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap()),
            iterations: 3,
            delta: T::zero(),
            weight_function: WeightFunction::Tricube,
            zero_weight_fallback: 0,
            robustness_method: RobustnessMethod::Bisquare,
            boundary_policy: BoundaryPolicy::default(),
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
            .delta(config.delta)
            .weight_function(config.weight_function)
            .zero_weight_fallback(config.zero_weight_fallback)
            .robustness_method(config.robustness_method)
            .boundary_policy(config.boundary_policy)
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

    /// Convert executor settings back to a `LoessConfig`.
    #[doc(hidden)]
    pub fn to_config(
        &self,
        fraction: Option<T>,
        tolerance: Option<T>,
        interval_method: Option<&IntervalMethod<T>>,
    ) -> LoessConfig<T> {
        LoessConfig {
            fraction: fraction.or(Some(self.fraction)),
            iterations: self.iterations,
            delta: self.delta,
            weight_function: self.weight_function,
            zero_weight_fallback: self.zero_weight_fallback,
            robustness_method: self.robustness_method,
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            auto_convergence: tolerance,
            return_variance: interval_method.cloned(),
            boundary_policy: self.boundary_policy,
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            custom_smooth_pass: self.custom_smooth_pass,
            custom_cv_pass: self.custom_cv_pass,
            custom_interval_pass: self.custom_interval_pass,
            custom_fit_pass: self.custom_fit_pass,
            parallel: self.parallel,
            backend: self.backend,
        }
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

    /// Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    /// Set the zero weight fallback policy flag.
    pub fn zero_weight_fallback(mut self, flag: u8) -> Self {
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

            // Run CV to find best fraction
            let (best_frac, scores) = if let Some(callback) = config.custom_cv_pass {
                callback(x, y, cv_fracs, cv_kind, &config)
            } else {
                cv_kind.run(x, y, cv_fracs, config.cv_seed, |tx, ty, f| {
                    executor.run(tx, ty, Some(f), None, None, None).smoothed
                })
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
    ) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let n = x.len();
        let eff_fraction = fraction.unwrap_or(self.fraction);

        // Handle global regression (fraction >= 1.0)
        if eff_fraction >= T::one() {
            let smoothed = GLSModel::global_ols(x, y);
            return ExecutorOutput {
                smoothed,
                std_errors: if confidence_method.is_some() {
                    Some(vec![T::zero(); n])
                } else {
                    None
                },
                iterations: None,
                used_fraction: eff_fraction,
                cv_scores: None,
                robustness_weights: vec![T::one(); n],
            };
        }

        // Calculate window size and prepare fitter
        let window_size = Window::calculate_span(n, eff_fraction);
        let fitter = LinearRegression;
        let target_iterations = max_iter.unwrap_or(self.iterations);

        // Handle boundary padding
        let (x_in, y_in, pad_len) = if self.boundary_policy != BoundaryPolicy::Extend {
            let (px, py) = apply_boundary_policy(x, y, window_size, self.boundary_policy);
            let pad = (px.len() - x.len()) / 2;
            (px, py, pad)
        } else {
            (x.to_vec(), y.to_vec(), 0)
        };

        let x_ref = &x_in;
        let y_ref = &y_in;

        // Run the iteration loop
        let (mut smoothed, mut std_errors, iterations, mut robustness_weights) = self
            .iteration_loop_with_callback(
                x_ref,
                y_ref,
                eff_fraction,
                window_size,
                target_iterations,
                self.delta,
                self.weight_function,
                self.zero_weight_fallback,
                &fitter,
                &self.robustness_method,
                confidence_method,
                tolerance,
                self.custom_smooth_pass,
                self.custom_interval_pass,
            );

        // Slice back to original range if padded
        if pad_len > 0 {
            Self::slice_results(
                n,
                pad_len,
                &mut smoothed,
                &mut std_errors,
                &mut robustness_weights,
            );
        }

        ExecutorOutput {
            smoothed,
            std_errors,
            iterations: if tolerance.is_some() {
                Some(iterations)
            } else {
                None
            },
            used_fraction: eff_fraction,
            cv_scores: None,
            robustness_weights,
        }
    }

    /// Perform the full LOESS iteration loop.
    #[allow(clippy::too_many_arguments)]
    pub fn iteration_loop_with_callback<Fitter>(
        &self,
        x: &[T],
        y: &[T],
        eff_fraction: T,
        window_size: usize,
        niter: usize,
        delta: T,
        weight_function: WeightFunction,
        zero_weight_flag: u8,
        fitter: &Fitter,
        robustness_updater: &RobustnessMethod,
        interval_method: Option<&IntervalMethod<T>>,
        convergence_tolerance: Option<T>,
        smooth_pass_fn: Option<SmoothPassFn<T>>,
        interval_pass_fn: Option<IntervalPassFn<T>>,
    ) -> (Vec<T>, Option<Vec<T>>, usize, Vec<T>)
    where
        Fitter: Regression<T> + ?Sized,
        T: Float + Debug + Send + Sync + 'static,
    {
        if self.custom_fit_pass.is_some() {
            let config = self.to_config(Some(eff_fraction), convergence_tolerance, interval_method);
            return (self.custom_fit_pass.unwrap())(x, y, &config);
        }

        let n = x.len();
        let mut buffers = IterationBuffers::allocate(n, convergence_tolerance.is_some());
        let mut iterations_performed = 0;

        // Copy initial y values to y_smooth
        buffers.y_smooth.copy_from_slice(y);

        // Smoothing iterations with robustness updates
        for iter in 0..=niter {
            iterations_performed = iter;

            // Swap buffers if checking convergence (save previous state)
            if convergence_tolerance.is_some() && iter > 0 {
                swap(&mut buffers.y_smooth, &mut buffers.y_prev);
            }

            // Perform smoothing pass
            if let Some(callback) = smooth_pass_fn {
                callback(
                    x,
                    y,
                    window_size,
                    delta,
                    iter > 0, // use_robustness
                    &buffers.robustness_weights,
                    &mut buffers.y_smooth,
                    weight_function,
                    zero_weight_flag,
                );
            } else {
                Self::smooth_pass(
                    x,
                    y,
                    window_size,
                    delta,
                    iter > 0, // use_robustness
                    &buffers.robustness_weights,
                    &mut buffers.y_smooth,
                    weight_function,
                    &mut buffers.weights,
                    zero_weight_flag,
                    fitter,
                );
            }

            // Check convergence if tolerance is provided (skip on first iteration)
            if let Some(tol) = convergence_tolerance {
                if iter > 0 && Self::check_convergence(&buffers.y_smooth, &buffers.y_prev, tol) {
                    break;
                }
            }

            // Update robustness weights for next iteration (skip last)
            if iter < niter {
                Self::update_robustness_weights(
                    y,
                    &buffers.y_smooth,
                    &mut buffers.residuals,
                    &mut buffers.robustness_weights,
                    robustness_updater,
                    &mut buffers.weights,
                );
            }
        }

        // Compute standard errors if requested
        let std_errors = interval_method.map(|im| {
            Self::compute_std_errors(
                x,
                y,
                &buffers.y_smooth,
                window_size,
                &buffers.robustness_weights,
                weight_function,
                im,
                interval_pass_fn,
            )
        });

        (
            buffers.y_smooth,
            std_errors,
            iterations_performed,
            buffers.robustness_weights,
        )
    }

    // ========================================================================
    // Main Algorithmic Logic
    // ========================================================================

    /// Perform a single smoothing pass over all points.
    #[allow(clippy::too_many_arguments)]
    pub fn smooth_pass<Fitter>(
        x: &[T],
        y: &[T],
        window_size: usize,
        delta: T,
        use_robustness: bool,
        robustness_weights: &[T],
        y_smooth: &mut [T],
        weight_function: WeightFunction,
        weights: &mut [T],
        zero_weight_flag: u8,
        fitter: &Fitter,
    ) where
        Fitter: Regression<T> + ?Sized,
    {
        let zero_weight_fallback = ZeroWeightFallback::from_u8(zero_weight_flag);

        // Fit first point
        let window = Self::fit_first_point(
            x,
            y,
            window_size,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            fitter,
            y_smooth,
        );

        // Fit remaining points with interpolation
        Self::fit_and_interpolate_remaining(
            x,
            y,
            delta,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            fitter,
            y_smooth,
            window,
        );
    }

    /// Compute standard errors for smoothed values.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_std_errors(
        x: &[T],
        y: &[T],
        y_smooth: &[T],
        window_size: usize,
        robustness_weights: &[T],
        weight_function: WeightFunction,
        interval_method: &IntervalMethod<T>,
        interval_pass_fn: Option<IntervalPassFn<T>>,
    ) -> Vec<T> {
        if let Some(callback) = interval_pass_fn {
            return callback(
                x,
                y,
                y_smooth,
                window_size,
                robustness_weights,
                weight_function,
                interval_method,
            );
        }

        let n = x.len();
        let mut se = vec![T::zero(); n];
        interval_method.compute_window_se(
            x,
            y,
            y_smooth,
            window_size,
            robustness_weights,
            &mut se,
            &|t| weight_function.compute_weight(t),
        );
        se
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Check convergence between current and previous smoothed values.
    pub fn check_convergence(y_smooth: &[T], y_prev: &[T], tolerance: T) -> bool {
        let max_change = y_smooth
            .iter()
            .zip(y_prev.iter())
            .fold(T::zero(), |maxv, (&current, &previous)| {
                T::max(maxv, (current - previous).abs())
            });

        max_change <= tolerance
    }

    /// Update robustness weights based on residuals.
    pub fn update_robustness_weights(
        y: &[T],
        y_smooth: &[T],
        residuals: &mut [T],
        robustness_weights: &mut [T],
        robustness_updater: &RobustnessMethod,
        scratch: &mut [T],
    ) {
        // Inline compute_residuals: residuals[i] = y[i] - y_smooth[i]
        for i in 0..y.len() {
            residuals[i] = y[i] - y_smooth[i];
        }
        robustness_updater.apply_robustness_weights(residuals, robustness_weights, scratch);
    }

    /// Helper to slice result buffers back to original data length when padding was used.
    fn slice_results(
        n: usize,
        pad_len: usize,
        smoothed: &mut Vec<T>,
        std_errors: &mut Option<Vec<T>>,
        robustness_weights: &mut Vec<T>,
    ) {
        smoothed.drain(0..pad_len);
        smoothed.truncate(n);

        if let Some(se) = std_errors.as_mut() {
            se.drain(0..pad_len);
            se.truncate(n);
        }

        robustness_weights.drain(0..pad_len);
        robustness_weights.truncate(n);
    }

    // ========================================================================
    // Specialized Fitting Functions
    // ========================================================================

    /// Fit the first point and initialize the smoothing window.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_single_point<Fitter>(
        x: &[T],
        y: &[T],
        idx: usize,
        window_size: usize,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        fitter: &Fitter,
    ) -> (T, Window)
    where
        Fitter: Regression<T> + ?Sized,
    {
        let n = x.len();
        let mut window = Window::initialize(idx, window_size, n);
        window.recenter(x, idx, n);

        let ctx = RegressionContext {
            x,
            y,
            idx,
            window,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
        };

        (fitter.fit(ctx).unwrap_or_else(|| y[idx]), window)
    }

    /// Fit the first point and initialize the smoothing window.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_first_point<Fitter>(
        x: &[T],
        y: &[T],
        window_size: usize,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        fitter: &Fitter,
        y_smooth: &mut [T],
    ) -> Window
    where
        Fitter: Regression<T> + ?Sized,
    {
        let (val, window) = Self::fit_single_point(
            x,
            y,
            0,
            window_size,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            fitter,
        );
        y_smooth[0] = val;
        window
    }

    /// Main fitting loop: iterate through remaining points with delta-skipping
    /// and linear interpolation.
    /// Uses binary search (partition_point) instead of linear scan to find
    /// the next anchor point. This reduces the overhead from O(n) to O(log n)
    /// per anchor, providing significant speedup when delta is large.
    #[allow(clippy::too_many_arguments)]
    fn fit_and_interpolate_remaining<Fitter>(
        x: &[T],
        y: &[T],
        delta: T,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        fitter: &Fitter,
        y_smooth: &mut [T],
        mut window: Window,
    ) where
        Fitter: Regression<T> + ?Sized,
    {
        let n = x.len();
        let mut last_fitted = 0usize;

        // Main loop: fit anchor points and interpolate between them
        while last_fitted < n - 1 {
            let cutpoint = x[last_fitted] + delta;

            // Binary search to find the first index where x > cutpoint
            // This is O(log n) instead of O(n) linear scan
            let next_idx =
                x[last_fitted + 1..].partition_point(|&xi| xi <= cutpoint) + last_fitted + 1;

            // Handle tied x-values: copy fitted value to all points with same x
            // Check the range [last_fitted+1, next_idx) for ties with last_fitted
            let mut tie_end = last_fitted;
            let x_last = x[last_fitted];
            for i in (last_fitted + 1)..next_idx.min(n) {
                if x[i] == x_last {
                    y_smooth[i] = y_smooth[last_fitted];
                    tie_end = i;
                } else {
                    break; // x is sorted, so no more ties
                }
            }
            if tie_end > last_fitted {
                last_fitted = tie_end;
            }

            // Determine current anchor point to fit
            // Either last point within delta range, or at minimum last_fitted+1
            let current = usize::max(next_idx.saturating_sub(1), last_fitted + 1).min(n - 1);

            // Check if we've made progress
            if current <= last_fitted {
                break;
            }

            // Update window to be centered around current point
            window.recenter(x, current, n);

            // Fit current point
            let ctx = RegressionContext {
                x,
                y,
                idx: current,
                window,
                use_robustness,
                robustness_weights,
                weights,
                weight_function,
                zero_weight_fallback,
            };

            y_smooth[current] = fitter.fit(ctx).unwrap_or_else(|| y[current]);

            // Linearly interpolate between last fitted and current
            interpolate_gap(x, y_smooth, last_fitted, current);
            last_fitted = current;
        }

        // Final interpolation to the end if necessary
        if last_fitted < n.saturating_sub(1) {
            // Fit the last point explicitly
            let final_idx = n - 1;
            window.recenter(x, final_idx, n);

            let ctx = RegressionContext {
                x,
                y,
                idx: final_idx,
                window,
                use_robustness,
                robustness_weights,
                weights,
                weight_function,
                zero_weight_fallback,
            };

            y_smooth[final_idx] = fitter.fit(ctx).unwrap_or_else(|| y[final_idx]);
            interpolate_gap(x, y_smooth, last_fitted, final_idx);
        }
    }
}
