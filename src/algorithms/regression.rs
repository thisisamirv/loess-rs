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
//! * **Algorithm**: Uses weighted least squares (WLS) for local polynomial regression.
//! * **Weights**: Computed from kernel functions, combined with robustness weights.
//! * **Fallback**: Implements policies for handling zero-weight or degenerate cases.
//! * **Generics**: Generic over `Float` types.
//! * **Unified**: Handles both 1D and nD cases via a generalized solver.
//!
//! ## Key concepts
//!
//! * **Local Weighted Regression**: Fits a polynomial model locally.
//! * **Weighted Least Squares**: Minimizes weighted sum of squared residuals.
//! * **Robustness Weights**: Downweights outliers (multiplied with kernel weights).
//!
//! ## Invariants
//!
//! * Weights are normalized for internal WLS calculations.
//! * Fitted values are always finite.
//!
//! ## Non-goals
//!
//! * This module does not manage the full smoothing iteration (handled by engine).
//! * This module does not compute diagnostics or intervals.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use num_traits::Float;

// Internal dependencies
use crate::math::kernel::WeightFunction;
use crate::math::neighborhood::Neighborhood;

// ============================================================================
// Polynomial Degree
// ============================================================================

/// Polynomial degree for local regression fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolynomialDegree {
    /// Degree 0: Local constant (weighted mean)
    Constant,

    /// Degree 1: Local linear regression (default)
    #[default]
    Linear,

    /// Degree 2: Local quadratic regression
    Quadratic,

    /// Degree 3: Local cubic regression
    Cubic,

    /// Degree 4: Local quartic regression
    Quartic,
}

impl PolynomialDegree {
    /// Get the numeric degree value.
    #[inline]
    pub const fn value(&self) -> usize {
        match self {
            PolynomialDegree::Constant => 0,
            PolynomialDegree::Linear => 1,
            PolynomialDegree::Quadratic => 2,
            PolynomialDegree::Cubic => 3,
            PolynomialDegree::Quartic => 4,
        }
    }

    /// Number of coefficients for 1D polynomial of this degree.
    #[inline]
    pub const fn num_coefficients_1d(&self) -> usize {
        self.value() + 1
    }

    /// Number of coefficients for nD polynomial of this degree.
    #[inline]
    pub const fn num_coefficients_nd(&self, dimensions: usize) -> usize {
        match self {
            PolynomialDegree::Constant => 1,
            PolynomialDegree::Linear => 1 + dimensions,
            PolynomialDegree::Quadratic => 1 + dimensions + (dimensions * (dimensions + 1)) / 2,
            PolynomialDegree::Cubic => {
                let n = dimensions;
                (n + 3) * (n + 2) * (n + 1) / 6
            }
            PolynomialDegree::Quartic => {
                let n = dimensions;
                (n + 4) * (n + 3) * (n + 2) * (n + 1) / 24
            }
        }
    }

    /// Build polynomial terms for a point relative to center.
    pub fn build_terms<T: Float>(&self, point: &[T], center: &[T], terms: &mut Vec<T>) {
        terms.clear();
        let d = point.len();

        // Intercept
        terms.push(T::one());

        if *self == PolynomialDegree::Constant {
            return;
        }

        // Precompute centered values to avoid repetitive substraction
        // We'll store them in a temporary stack buffer if small, but here vec is acceptable.
        // Or just map.
        let centered: Vec<T> = point
            .iter()
            .zip(center.iter())
            .map(|(&p, &c)| p - c)
            .collect();

        // Linear terms
        for &val in &centered {
            terms.push(val);
        }

        if *self == PolynomialDegree::Linear {
            return;
        }

        // Quadratic terms
        for i in 0..d {
            for j in i..d {
                terms.push(centered[i] * centered[j]);
            }
        }

        if *self == PolynomialDegree::Quadratic {
            return;
        }

        // Cubic terms
        for i in 0..d {
            for j in i..d {
                for k in j..d {
                    terms.push(centered[i] * centered[j] * centered[k]);
                }
            }
        }

        if *self == PolynomialDegree::Cubic {
            return;
        }

        // Quartic terms
        for i in 0..d {
            for j in i..d {
                for k in j..d {
                    for l in k..d {
                        terms.push(centered[i] * centered[j] * centered[k] * centered[l]);
                    }
                }
            }
        }
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
    /// Create from u8 flag.
    #[inline]
    pub fn from_u8(flag: u8) -> Self {
        match flag {
            0 => ZeroWeightFallback::UseLocalMean,
            1 => ZeroWeightFallback::ReturnOriginal,
            2 => ZeroWeightFallback::ReturnNone,
            _ => ZeroWeightFallback::UseLocalMean,
        }
    }

    /// Convert to u8 flag.
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
// Regression Context
// ============================================================================

/// Context containing all data needed to fit a single point (unified 1D/nD).
pub struct RegressionContext<'a, T: Float> {
    /// Flattened array of predictor values.
    /// For 1D: [x₁, x₂, ...]
    /// For nD: [x₁₁, x₁₂, ..., x₂₁, x₂₂, ...]
    pub x: &'a [T],

    /// Number of dimensions per point.
    pub dimensions: usize,

    /// Slice of response values.
    pub y: &'a [T],

    /// Index of the current query point (if using a point from x).
    pub query_idx: usize,

    /// Explicit query point (if not using query_idx).
    /// If None, assumes query is `x[query_idx]`.
    pub query_point: Option<&'a [T]>,

    /// Neighborhood of k-nearest neighbors.
    pub neighborhood: &'a Neighborhood<T>,

    /// Whether to use robustness weights.
    pub use_robustness: bool,

    /// Robustness weights (all 1.0 if not using).
    pub robustness_weights: &'a [T],

    /// Weight function (kernel).
    pub weight_function: WeightFunction,

    /// Zero-weight fallback policy.
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Polynomial degree.
    pub polynomial_degree: PolynomialDegree,

    /// Whether to compute and return leverage.
    pub compute_leverage: bool,
}

// ============================================================================
// WLS Fitting (Unified)
// ============================================================================

impl<'a, T: Float> RegressionContext<'a, T> {
    /// Returns the (predicted value, leverage) at the query point.
    pub fn fit(&self) -> Option<(T, T)> {
        let n_neighbors = self.neighborhood.len();
        if n_neighbors == 0 {
            return None;
        }

        let d = self.dimensions;
        let n_coeffs = self.polynomial_degree.num_coefficients_nd(d);
        let max_distance = self.neighborhood.max_distance;

        // Handle zero bandwidth case (all neighbors at same location)
        if max_distance <= T::epsilon() {
            let (val, sum_w) = self.weighted_mean_and_sum();
            let leverage = if sum_w > T::epsilon() {
                T::one() / sum_w
            } else {
                T::zero()
            };
            return Some((val, leverage));
        }

        // Compute final weights (kernel × robustness)
        let mut weights = Vec::with_capacity(n_neighbors);
        for i in 0..n_neighbors {
            let neighbor_idx = self.neighborhood.indices[i];
            let dist = self.neighborhood.distances[i];

            // Normalize distance by bandwidth
            let u = dist / max_distance;

            // Kernel weight
            let kernel_w = self.weight_function.compute_weight(u);

            // Combined weight
            let w = if self.use_robustness {
                kernel_w * self.robustness_weights[neighbor_idx]
            } else {
                kernel_w
            };

            weights.push(w);
        }

        // Check numerical stability of weights
        let weight_sum: T = weights.iter().copied().fold(T::zero(), |a, b| a + b);
        if weight_sum <= T::epsilon() {
            return self.handle_zero_weights();
        }

        // Get query point coordinates
        let query_point = if let Some(qp) = self.query_point {
            qp
        } else {
            let query_offset = self.query_idx * d;
            &self.x[query_offset..query_offset + d]
        };

        // For constant degree, just compute weighted mean
        if self.polynomial_degree == PolynomialDegree::Constant {
            let (val, sum_w) = self.weighted_mean_and_sum();
            let leverage = if sum_w > T::epsilon() {
                T::one() / sum_w
            } else {
                T::zero()
            };
            return Some((val, leverage));
        }

        // Build and solve WLS system
        self.fit_polynomial_wls(&weights, query_point, n_coeffs)
    }

    /// Handle zero weight cases using fallback policy.
    fn handle_zero_weights(&self) -> Option<(T, T)> {
        match self.zero_weight_fallback {
            ZeroWeightFallback::UseLocalMean => {
                // Unweighted mean of the neighborhood y-values
                let n = T::from(self.neighborhood.len()).unwrap();
                let sum_y = self
                    .neighborhood
                    .indices
                    .iter()
                    .map(|&i| self.y[i])
                    .fold(T::zero(), |a, b| a + b);
                Some((sum_y / n, T::zero()))
            }
            ZeroWeightFallback::ReturnOriginal => {
                // If we are fitting a point in the set (idx valid)
                Some((self.y[self.query_idx], T::zero()))
            }
            ZeroWeightFallback::ReturnNone => None,
        }
    }

    /// Compute weighted mean and sum of weights.
    fn weighted_mean_and_sum(&self) -> (T, T) {
        let mut sum_wy = T::zero();
        let mut sum_w = T::zero();

        let max_dist = self.neighborhood.max_distance;
        let bandwidth = if max_dist > T::epsilon() {
            max_dist
        } else {
            T::one()
        };

        for i in 0..self.neighborhood.len() {
            let idx = self.neighborhood.indices[i];
            let dist = self.neighborhood.distances[i];

            let u = dist / bandwidth;
            let kernel_w = self.weight_function.compute_weight(u);

            let w = if self.use_robustness {
                kernel_w * self.robustness_weights[idx]
            } else {
                kernel_w
            };

            sum_wy = sum_wy + w * self.y[idx];
            sum_w = sum_w + w;
        }

        let val = if sum_w > T::epsilon() {
            sum_wy / sum_w
        } else {
            // Fallback to simple mean if weights collapsed
            let n = T::from(self.neighborhood.len()).unwrap();
            self.neighborhood
                .indices
                .iter()
                .map(|&i| self.y[i])
                .fold(T::zero(), |a, b| a + b)
                / n
        };
        (val, sum_w)
    }

    /// Fit polynomial via weighted least squares.
    fn fit_polynomial_wls(
        &self,
        weights: &[T],
        query_point: &[T],
        n_coeffs: usize,
    ) -> Option<(T, T)> {
        let n_neighbors = self.neighborhood.len();
        let d = self.dimensions;

        // Need at least as many neighbors as coefficients
        if n_neighbors < n_coeffs {
            let (val, sum_w) = self.weighted_mean_and_sum();
            return Some((
                val,
                if sum_w > T::epsilon() {
                    T::one() / sum_w
                } else {
                    T::zero()
                },
            ));
        }

        // Build XᵀWX and XᵀWy
        let mut xtw_x = vec![T::zero(); n_coeffs * n_coeffs];
        let mut xtw_y = vec![T::zero(); n_coeffs];
        let mut terms = Vec::with_capacity(n_coeffs);

        for (i, &neighbor_idx) in self.neighborhood.indices.iter().enumerate() {
            let w = weights[i];
            if w <= T::epsilon() {
                continue;
            }

            let offset = neighbor_idx * d;
            let neighbor_point = &self.x[offset..offset + d];
            let y_val = self.y[neighbor_idx];

            self.polynomial_degree
                .build_terms(neighbor_point, query_point, &mut terms);

            for j in 0..n_coeffs {
                xtw_y[j] = xtw_y[j] + w * terms[j] * y_val;
                for k in j..n_coeffs {
                    let val = w * terms[j] * terms[k];
                    xtw_x[j * n_coeffs + k] = xtw_x[j * n_coeffs + k] + val;
                    if k != j {
                        xtw_x[k * n_coeffs + j] = xtw_x[k * n_coeffs + j] + val;
                    }
                }
            }
        }

        // Solve linear system
        let beta = LinearSolver::solve_symmetric(&xtw_x, &xtw_y, n_coeffs)?;

        let leverage = if self.compute_leverage {
            let mut e1 = vec![T::zero(); n_coeffs];
            e1[0] = T::one();
            let v = LinearSolver::solve_symmetric(&xtw_x, &e1, n_coeffs)?;
            v[0]
        } else {
            T::zero()
        };

        Some((beta[0], leverage))
    }
}

// ============================================================================
// Linear Solver
// ============================================================================

/// Helper struct for solving linear systems (e.g., Cholesky decomposition).
pub struct LinearSolver;

impl LinearSolver {
    /// Solve Ax = b where A is symmetric positive definite
    pub fn solve_symmetric<T: Float>(a: &[T], b: &[T], n: usize) -> Option<Vec<T>> {
        let l = Self::cholesky_decompose(a, n)?;

        // Forward Ly = b
        let mut y = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum = sum - l[i * n + j] * y[j];
            }
            if l[i * n + i].abs() <= T::epsilon() {
                return None;
            }
            y[i] = sum / l[i * n + i];
        }

        // Backward Lᵀx = y
        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum = sum - l[j * n + i] * x[j];
            }
            if l[i * n + i].abs() <= T::epsilon() {
                return None;
            }
            x[i] = sum / l[i * n + i];
        }
        Some(x)
    }

    /// Cholesky decomposition A = LLᵀ
    pub fn cholesky_decompose<T: Float>(a: &[T], n: usize) -> Option<Vec<T>> {
        let mut l = vec![T::zero(); n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                for k in 0..j {
                    sum = sum - l[i * n + k] * l[j * n + k];
                }

                if i == j {
                    if sum <= T::zero() {
                        // Regularization
                        let reg = T::from(1e-10).unwrap();
                        sum = sum + reg;
                        if sum <= T::zero() {
                            return None;
                        }
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    let diag = l[j * n + j];
                    if diag.abs() <= T::epsilon() {
                        return None;
                    }
                    l[i * n + j] = sum / diag;
                }
            }
        }
        Some(l)
    }
}
