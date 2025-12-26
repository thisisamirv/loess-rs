//! Local weighted regression for LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the core fitting algorithm for LOESS: local weighted
//! linear regression. At each point, a linear model is fit using nearby points
//! with weights determined by a kernel function.
//!
//! ## Design notes
//!
//! * **Algorithm**: Uses weighted least squares (WLS) for local linear regression.
//! * **Weights**: Computed from kernel functions, combined with robustness weights.
//! * **Fallback**: Implements policies for handling zero-weight or degenerate cases.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Local Weighted Regression**: Fits a linear model `y = beta_0 + beta_1 * x` locally.
//! * **Weighted Least Squares**: Minimizes weighted sum of squared residuals.
//! * **Robustness Weights**: Downweights outliers (multiplied with kernel weights).
//! * **Zero-Weight Fallback**: Handles numerical stability issues (e.g., use local mean).
//!
//! ## Invariants
//!
//! * Window radius must be positive.
//! * Weights are normalized for internal WLS calculations.
//! * Fitted values are always finite.
//!
//! ## Non-goals
//!
//! * This module does not compute the windows (done by window module).
//! * This module does not compute robustness weights (done by robustness module).
//! * This module does not perform higher-degree polynomial regression.
//! * This module does not validate input data.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use num_traits::Float;

// Internal dependencies
use crate::math::kernel::WeightFunction;
use crate::primitives::window::Window;

// ============================================================================
// Regression Context
// ============================================================================

/// Context containing all data needed to fit a single point.
pub struct RegressionContext<'a, T: Float> {
    /// Slice of x-values (independent variable)
    pub x: &'a [T],

    /// Slice of y-values (dependent variable)
    pub y: &'a [T],

    /// Index of the point to fit
    pub idx: usize,

    /// Window for the local fit (defines neighborhood)
    pub window: Window,

    /// Whether to use robustness weights
    pub use_robustness: bool,

    /// Slice of robustness weights (all 1.0 if not using robustness)
    pub robustness_weights: &'a [T],

    /// Mutable slice of weights to be used in fitting
    pub weights: &'a mut [T],

    /// Weight function (kernel)
    pub weight_function: WeightFunction,

    /// Zero-weight fallback policy
    pub zero_weight_fallback: ZeroWeightFallback,
}

// ============================================================================
// Regression Trait
// ============================================================================

/// Trait for fitting a local regression at a single point.
pub trait Regression<T: Float>: Sync + Send {
    /// Fit the point specified in the context.
    fn fit(&self, context: RegressionContext<T>) -> Option<T>;
}

// ============================================================================
// Linear Regression Implementation
// ============================================================================

/// Standard local linear regression fitter.
///
/// # Special cases
///
/// * **Zero window radius**: Falls back to weighted average
/// * **Zero weight sum**: Applies zero-weight fallback policy
/// * **Zero variance**: Returns horizontal line at weighted mean
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearRegression;

impl<T: Float> Regression<T> for LinearRegression {
    fn fit(&self, context: RegressionContext<T>) -> Option<T> {
        let n = context.x.len();

        // Defensive bounds check
        if context.idx >= n {
            return None;
        }

        // Validate window bounds
        if context.window.left >= n || context.window.right >= n {
            return None;
        }

        let x_current = context.x[context.idx];

        // Compute window radius
        let window_radius = context.window.max_distance(context.x, x_current);

        if window_radius <= T::zero() {
            // Degenerate window (all x-values identical): fall back to weighted average.
            // We use uniform weights or robustness weights if enabled.
            let mut sum_w = T::zero();
            let mut sum_wy = T::zero();
            for j in context.window.left..=context.window.right {
                let w = if context.use_robustness {
                    context.robustness_weights[j]
                } else {
                    T::one()
                };
                sum_w = sum_w + w;
                sum_wy = sum_wy + w * context.y[j];
            }

            if sum_w > T::zero() {
                return Some(sum_wy / sum_w);
            } else {
                // If weight sum is zero, apply standard fallback (local mean or original)
                return match context.zero_weight_fallback {
                    ZeroWeightFallback::UseLocalMean => {
                        let window_size = context.window.right - context.window.left + 1;
                        let mean = context.y[context.window.left..=context.window.right]
                            .iter()
                            .copied()
                            .fold(T::zero(), |acc, v| acc + v)
                            / T::from(window_size as f64).unwrap_or(T::one());
                        Some(mean)
                    }
                    ZeroWeightFallback::ReturnOriginal => Some(context.y[context.idx]),
                    ZeroWeightFallback::ReturnNone => None,
                };
            }
        }

        let weight_params = WeightParams::new(x_current, window_radius, context.use_robustness);

        // Compute raw kernel weights (unnormalized, without robustness)
        let (mut weight_sum, rightmost_idx) = context.weight_function.compute_window_weights(
            context.x,
            context.window.left,
            context.window.right,
            weight_params.x_current,
            weight_params.window_radius,
            weight_params.h1,
            weight_params.h9,
            context.weights,
        );

        // Apply robustness weights if needed and compute final sum
        if context.use_robustness {
            weight_sum = T::zero();
            for j in context.window.left..=rightmost_idx {
                let w_k = context.weights[j];
                if w_k > T::zero() {
                    let w_robust = context.robustness_weights[j];
                    let w_final = w_k * w_robust;
                    context.weights[j] = w_final;
                    weight_sum = weight_sum + w_final;
                }
            }
        }

        if weight_sum <= T::zero() {
            // Handle zero-weight case according to fallback policy
            match context.zero_weight_fallback {
                ZeroWeightFallback::UseLocalMean => {
                    let window_size = context.window.right - context.window.left + 1;
                    let cnt = T::from(window_size).unwrap_or(T::one());
                    let mean = context.y[context.window.left..=context.window.right]
                        .iter()
                        .copied()
                        .fold(T::zero(), |acc, v| acc + v)
                        / cnt;
                    return Some(mean);
                }
                ZeroWeightFallback::ReturnOriginal => {
                    return Some(context.y[context.idx]);
                }
                ZeroWeightFallback::ReturnNone => {
                    return None;
                }
            }
        }

        // Perform weighted least squares regression
        // We pass the weight_sum to avoid re-summing it inside GLSModel::fit_wls
        Some(GLSModel::local_wls_with_sum(
            context.x,
            context.y,
            context.weights,
            context.window.left,
            rightmost_idx,
            x_current,
            window_radius,
            weight_sum,
        ))
    }
}

// ============================================================================
// Zero-Weight Fallback Policy
// ============================================================================

/// Policy for handling cases where all weights are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZeroWeightFallback {
    /// Use local mean (default).
    #[default]
    UseLocalMean,

    /// Return the original y-value.
    ReturnOriginal,

    /// Return None (propagate failure).
    ReturnNone,
}

impl ZeroWeightFallback {
    /// Create from u8 flag for backward compatibility.
    #[inline]
    pub fn from_u8(flag: u8) -> Self {
        match flag {
            0 => ZeroWeightFallback::UseLocalMean,
            1 => ZeroWeightFallback::ReturnOriginal,
            2 => ZeroWeightFallback::ReturnNone,
            _ => ZeroWeightFallback::UseLocalMean, // Default for unknown values
        }
    }

    /// Convert to u8 flag for backward compatibility.
    #[inline]
    pub fn to_u8(self) -> u8 {
        match self {
            ZeroWeightFallback::UseLocalMean => 0,
            ZeroWeightFallback::ReturnOriginal => 1,
            ZeroWeightFallback::ReturnNone => 2,
        }
    }
}

// ============================================================================
// Weight Parameters
// ============================================================================

/// Parameters for weight computation.
///
/// # Implementation Note
///
/// The thresholds `h1` and `h9` are implementation optimizations:
/// * `h1 = 0.001 × window_radius`: Points closer than this get weight 1.0
///   (avoids near-zero distance numerical issues)
/// * `h9 = 0.999 × window_radius`: Points farther than this get weight 0.0
///   (early termination for efficiency in sorted arrays)
///
/// These are not formal LOESS parameters but implementation details for
/// numerical stability and performance.
pub struct WeightParams<T: Float> {
    /// Current x-value being fitted
    pub x_current: T,

    /// Window radius - defines the scale of the local fit
    pub window_radius: T,

    /// Near-threshold: points closer than this get weight 1.0.
    /// Internal optimization: 0.001 × radius.
    pub h1: T,

    /// Far-threshold: points farther than this get weight 0.0.
    /// Internal optimization: 0.999 × radius.
    pub h9: T,
}

impl<T: Float> WeightParams<T> {
    /// Construct WeightParams with validated window radius.
    pub fn new(x_current: T, window_radius: T, _use_robustness: bool) -> Self {
        debug_assert!(
            window_radius > T::zero(),
            "WeightParams::new: window_radius must be positive"
        );

        // In release builds avoid panic: clamp tiny/zero radius to small epsilon
        let radius = if window_radius > T::zero() {
            window_radius
        } else {
            // Small absolute fallback for numerical stability
            T::from(1e-12).unwrap()
        };

        let h1 = T::from(0.001).unwrap() * radius;
        let h9 = T::from(0.999).unwrap() * radius;

        Self {
            x_current,
            window_radius: radius,
            h1,
            h9,
        }
    }
}

// ============================================================================
// Generalized Least Squares Model
// ============================================================================

/// Generalized Least Squares (GLS) model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GLSModel<T: Float> {
    /// Slope (beta_1)
    pub slope: T,

    /// Intercept (beta_0)
    pub intercept: T,

    /// Weighted mean of x-values
    pub x_mean: T,

    /// Weighted mean of y-values
    pub y_mean: T,
}

impl<T: Float> GLSModel<T> {
    /// Predict y-value for a given x.
    fn predict(&self, x: T) -> T {
        self.intercept + self.slope * x
    }

    /// Fit Ordinary Least Squares (OLS) regression.
    fn fit_ols(x: &[T], y: &[T]) -> Self
    where
        T: Debug,
    {
        let n = x.len();
        if n == 0 {
            return GLSModel {
                slope: T::zero(),
                intercept: T::zero(),
                x_mean: T::zero(),
                y_mean: T::zero(),
            };
        }

        let n_t = T::from(n).unwrap_or(T::one());

        let mut sum_x = T::zero();
        let mut sum_y = T::zero();

        for i in 0..n {
            sum_x = sum_x + x[i];
            sum_y = sum_y + y[i];
        }

        let x_mean = sum_x / n_t;
        let y_mean = sum_y / n_t;

        let mut variance = T::zero();
        let mut covariance = T::zero();

        for i in 0..n {
            let dx = x[i] - x_mean;
            let dy = y[i] - y_mean;
            variance = variance + dx * dx;
            covariance = covariance + dx * dy;
        }

        let tol = T::from(1e-12).unwrap();
        if variance <= tol {
            // Fallback: horizontal line at y_mean
            return GLSModel {
                slope: T::zero(),
                intercept: y_mean,
                x_mean,
                y_mean,
            };
        }

        let slope = covariance / variance;
        let intercept = y_mean - slope * x_mean;

        GLSModel {
            slope,
            intercept,
            x_mean,
            y_mean,
        }
    }

    /// Compute global OLS linear fit for the full dataset.
    pub fn global_ols(x: &[T], y: &[T]) -> Vec<T>
    where
        T: Debug,
    {
        let model = Self::fit_ols(x, y);
        x.iter().map(|&xi| model.predict(xi)).collect()
    }

    /// Fit Weighted Least Squares (WLS) regression.
    fn fit_wls(x: &[T], y: &[T], weights: &[T], window_radius: T, sum_w: T) -> GLSModel<T> {
        let n = x.len();
        if n == 0 || sum_w <= T::zero() {
            return GLSModel {
                slope: T::zero(),
                intercept: T::zero(),
                x_mean: T::zero(),
                y_mean: T::zero(),
            };
        }

        // Compute weighted means
        let mut x_mean = T::zero();
        let mut y_mean = T::zero();
        for i in 0..n {
            let w = weights[i];
            x_mean = x_mean + w * x[i];
            y_mean = y_mean + w * y[i];
        }

        x_mean = x_mean / sum_w;
        y_mean = y_mean / sum_w;

        // Degenerate window_radius: simple weighted average
        if window_radius <= T::zero() {
            return GLSModel {
                slope: T::zero(),
                intercept: y_mean,
                x_mean,
                y_mean,
            };
        }

        // Compute weighted covariance and variance
        let mut variance = T::zero(); // sum(w * (x - x_mean)^2)
        let mut covariance = T::zero(); // sum(w * (x - x_mean) * (y - y_mean))

        for i in 0..n {
            let w = weights[i] / sum_w; // Implicitly use normalized weight for variance/covariance
            let dx = x[i] - x_mean;
            let dy = y[i] - y_mean;
            variance = variance + w * dx * dx;
            covariance = covariance + w * dx * dy;
        }

        // Numerical stability check for variance
        let abs_tol = T::from(1e-7).unwrap();
        let rel_tol = T::epsilon() * window_radius * window_radius;
        let tol = abs_tol.max(rel_tol);

        if variance <= tol {
            return GLSModel {
                slope: T::zero(),
                intercept: y_mean,
                x_mean,
                y_mean,
            };
        }

        let slope = covariance / variance;
        let intercept = y_mean - slope * x_mean;

        GLSModel {
            slope,
            intercept,
            x_mean,
            y_mean,
        }
    }

    /// Weighted linear regression evaluated at a specific point with precomputed weight sum.
    #[allow(clippy::too_many_arguments)]
    pub fn local_wls_with_sum(
        x: &[T],
        y: &[T],
        weights: &[T],
        left: usize,
        right: usize,
        x_current: T,
        window_radius: T,
        sum_w: T,
    ) -> T {
        let window_x = &x[left..=right];
        let window_y = &y[left..=right];
        let window_weights = &weights[left..=right];

        let model = Self::fit_wls(window_x, window_y, window_weights, window_radius, sum_w);
        model.predict(x_current)
    }
}
