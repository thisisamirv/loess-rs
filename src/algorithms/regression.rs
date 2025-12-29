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
use core::marker::PhantomData;
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
    pub fn build_terms<T: Float>(&self, point: &[T], center: &[T], terms: &mut [T]) -> usize {
        let d = point.len();
        let degree = self.value();

        // Intercept
        terms[0] = T::one();
        if degree == 0 {
            return 1;
        }

        // Special case 1D for speed
        if d == 1 {
            let x = point[0] - center[0];
            terms[1] = x;
            if degree == 1 {
                return 2;
            }
            terms[2] = x * x;
            if degree == 2 {
                return 3;
            }
            terms[3] = x * x * x;
            if degree == 3 {
                return 4;
            }
            terms[4] = x * x * x * x;
            return 5;
        }

        // Special case 2D for speed
        if d == 2 {
            let x = point[0] - center[0];
            let y = point[1] - center[1];
            terms[1] = x;
            terms[2] = y;
            if degree == 1 {
                return 3;
            }

            // Quadratic: x^2, xy, y^2
            terms[3] = x * x;
            terms[4] = x * y;
            terms[5] = y * y;
            if degree == 2 {
                return 6;
            }

            // Cubic: x^3, x^2y, xy^2, y^3
            terms[6] = x * x * x;
            terms[7] = x * x * y;
            terms[8] = x * y * y;
            terms[9] = y * y * y;
            if degree == 3 {
                return 10;
            }

            // Quartic: x^4, x^3y, x^2y^2, xy^3, y^4
            terms[10] = x * x * x * x;
            terms[11] = x * x * x * y;
            terms[12] = x * x * y * y;
            terms[13] = x * y * y * y;
            terms[14] = y * y * y * y;
            return 15;
        }

        // General nD case
        let mut count = 1;

        // Linear terms (and store centered values for higher degrees)
        for i in 0..d {
            let val = point[i] - center[i];
            terms[count] = val;
            count += 1;
        }

        if degree == 1 {
            return count;
        }

        // Quadratic
        for i in 0..d {
            for j in i..d {
                terms[count] = terms[1 + i] * terms[1 + j];
                count += 1;
            }
        }
        if degree == 2 {
            return count;
        }

        // Cubic
        for i in 0..d {
            for j in i..d {
                for k in j..d {
                    terms[count] = terms[1 + i] * terms[1 + j] * terms[1 + k];
                    count += 1;
                }
            }
        }
        if degree == 3 {
            return count;
        }

        // Quartic
        for i in 0..d {
            for j in i..d {
                for k in j..d {
                    for l in k..d {
                        terms[count] = terms[1 + i] * terms[1 + j] * terms[1 + k] * terms[1 + l];
                        count += 1;
                    }
                }
            }
        }
        count
    }
}

// ============================================================================
// Term Generators (Strategy Pattern)
// ============================================================================

/// Trait for generating polynomial terms for a given point.
/// This abstracts the dimensionality and degree logic from the solver.
pub trait TermGenerator<T: Float> {
    /// Returns the number of coefficients (terms) generated.
    fn n_coeffs(&self) -> usize;

    /// Generates terms for a single point relative to the query point.
    /// Writes terms into `out`, which must have length >= `n_coeffs()`.
    ///
    /// `point`: The predictor values of the neighbor.
    /// `query`: The predictor values of the query point.
    /// `out`: The output buffer for terms.
    fn generate(&self, point: &[T], query: &[T], out: &mut [T]);
}

/// Generator for N-dimensional generic polynomial terms.
pub struct GenericTermGenerator<'a, T: Float> {
    degree: PolynomialDegree,
    _d: usize,
    n_coeffs: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: Float> GenericTermGenerator<'a, T> {
    /// Create a new generic term generator.
    pub fn new(degree: PolynomialDegree, d: usize, n_coeffs: usize) -> Self {
        Self {
            degree,
            _d: d,
            n_coeffs,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Float> TermGenerator<T> for GenericTermGenerator<'a, T> {
    #[inline(always)]
    fn n_coeffs(&self) -> usize {
        self.n_coeffs
    }

    #[inline(always)]
    fn generate(&self, point: &[T], query: &[T], out: &mut [T]) {
        self.degree.build_terms(point, query, out);
    }
}

/// Specialized Generator for 1D Linear (Terms: 1, x).
pub struct Linear1DTermGenerator;
impl<T: Float> TermGenerator<T> for Linear1DTermGenerator {
    #[inline(always)]
    fn n_coeffs(&self) -> usize {
        2
    }

    #[inline(always)]
    fn generate(&self, point: &[T], query: &[T], out: &mut [T]) {
        out[0] = T::one();
        out[1] = point[0] - query[0];
    }
}

/// Specialized Generator for 2D Linear (Terms: 1, x, y).
pub struct Linear2DTermGenerator;
impl<T: Float> TermGenerator<T> for Linear2DTermGenerator {
    #[inline(always)]
    fn n_coeffs(&self) -> usize {
        3
    }

    #[inline(always)]
    fn generate(&self, point: &[T], query: &[T], out: &mut [T]) {
        out[0] = T::one();
        out[1] = point[0] - query[0];
        out[2] = point[1] - query[1];
    }
}

/// Specialized Generator for 3D Linear (Terms: 1, x, y, z).
pub struct Linear3DTermGenerator;
impl<T: Float> TermGenerator<T> for Linear3DTermGenerator {
    #[inline(always)]
    fn n_coeffs(&self) -> usize {
        4
    }

    #[inline(always)]
    fn generate(&self, point: &[T], query: &[T], out: &mut [T]) {
        out[0] = T::one();
        out[1] = point[0] - query[0];
        out[2] = point[1] - query[1];
        out[3] = point[2] - query[2];
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

/// Persistent buffers for local regression to avoid allocations.
pub struct FittingBuffer<T: Float> {
    /// Weights for each neighbor.
    pub weights: Vec<T>,
    /// Normal matrix X'WX.
    pub xtw_x: Vec<T>,
    /// Normal vector X'WY.
    pub xtw_y: Vec<T>,
}

impl<T: Float> FittingBuffer<T> {
    /// Create a new fitting buffer with estimated capacities.
    pub fn new(k: usize, n_coeffs: usize) -> Self {
        Self {
            weights: Vec::with_capacity(k),
            xtw_x: Vec::with_capacity(n_coeffs * n_coeffs),
            xtw_y: Vec::with_capacity(n_coeffs),
        }
    }
}

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
    /// Optional persistent buffer for reuse.
    pub buffer: Option<&'a mut FittingBuffer<T>>,
}

// ============================================================================
// WLS Fitting (Unified)
// ============================================================================

impl<'a, T: Float + 'static> RegressionContext<'a, T> {
    /// Returns the (predicted value, leverage) at the query point.
    pub fn fit(&mut self) -> Option<(T, T)> {
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

        let mut buffer = self.buffer.take();
        let result = if let Some(ref mut buf) = buffer {
            buf.weights.clear();
            let weights = &mut buf.weights;

            for i in 0..n_neighbors {
                let neighbor_idx = self.neighborhood.indices[i];
                let dist = self.neighborhood.distances[i];
                let u = (dist / max_distance).sqrt();
                let kernel_w = self.weight_function.compute_weight(u);
                let w = if self.use_robustness {
                    (kernel_w * self.robustness_weights[neighbor_idx]).sqrt()
                } else {
                    kernel_w.sqrt()
                };
                weights.push(w);
            }

            // Check numerical stability of weights
            let weight_sum: T = weights.iter().copied().fold(T::zero(), |a, b| a + b);
            if weight_sum <= T::epsilon() {
                self.buffer = buffer;
                return self.handle_zero_weights();
            }

            buf.xtw_x.resize(n_coeffs * n_coeffs, T::zero());
            buf.xtw_y.resize(n_coeffs, T::zero());

            self.fit_polynomial_wls_internal(
                weights,
                query_point,
                n_coeffs,
                &mut buf.xtw_x,
                &mut buf.xtw_y,
            )
        } else {
            let mut weights = Vec::with_capacity(n_neighbors);
            for i in 0..n_neighbors {
                let neighbor_idx = self.neighborhood.indices[i];
                let dist = self.neighborhood.distances[i];
                let u = (dist / max_distance).sqrt();
                let kernel_w = self.weight_function.compute_weight(u);
                let w = if self.use_robustness {
                    (kernel_w * self.robustness_weights[neighbor_idx]).sqrt()
                } else {
                    kernel_w.sqrt()
                };
                weights.push(w);
            }

            // Check numerical stability of weights
            let weight_sum: T = weights.iter().copied().fold(T::zero(), |a, b| a + b);
            if weight_sum <= T::epsilon() {
                self.buffer = buffer;
                return self.handle_zero_weights();
            }

            let mut xtw_x = vec![T::zero(); n_coeffs * n_coeffs];
            let mut xtw_y = vec![T::zero(); n_coeffs];

            self.fit_polynomial_wls_internal(
                &weights,
                query_point,
                n_coeffs,
                &mut xtw_x,
                &mut xtw_y,
            )
        };
        self.buffer = buffer;
        result
    }

    /// Returns all polynomial coefficients [value, d/dx1, d/dx2, ..., d/dxd] at the query point.
    /// Used for Hermite interpolation which requires derivatives.
    pub fn fit_with_coefficients(&mut self) -> Option<Vec<T>> {
        let n_neighbors = self.neighborhood.len();
        if n_neighbors == 0 {
            return None;
        }

        let d = self.dimensions;
        let n_coeffs = self.polynomial_degree.num_coefficients_nd(d);
        let max_distance = self.neighborhood.max_distance;

        // Handle zero bandwidth or constant degree - return [value, 0, 0, ...]
        if max_distance <= T::epsilon() || self.polynomial_degree == PolynomialDegree::Constant {
            let (val, _) = self.weighted_mean_and_sum();
            let mut coeffs = vec![T::zero(); d + 1];
            coeffs[0] = val;
            return Some(coeffs);
        }

        let query_point = if let Some(qp) = self.query_point {
            qp
        } else {
            let query_offset = self.query_idx * d;
            &self.x[query_offset..query_offset + d]
        };

        // Build normal equations and solve
        let mut xtw_x = vec![T::zero(); n_coeffs * n_coeffs];
        let mut xtw_y = vec![T::zero(); n_coeffs];

        // Compute weights
        let mut weights = Vec::with_capacity(n_neighbors);
        for i in 0..n_neighbors {
            let neighbor_idx = self.neighborhood.indices[i];
            let dist = self.neighborhood.distances[i];
            let u = (dist / max_distance).sqrt();
            let kernel_w = self.weight_function.compute_weight(u);
            let w = if self.use_robustness {
                (kernel_w * self.robustness_weights[neighbor_idx]).sqrt()
            } else {
                kernel_w.sqrt()
            };
            weights.push(w);
        }

        // Accumulate normal equations based on polynomial degree and dimensions
        match (self.dimensions, self.polynomial_degree) {
            (1, PolynomialDegree::Linear) => {
                self.accumulate_normal_equations(
                    &weights,
                    Linear1DTermGenerator,
                    query_point,
                    &mut xtw_x,
                    &mut xtw_y,
                );
            }
            (2, PolynomialDegree::Linear) => {
                self.accumulate_normal_equations(
                    &weights,
                    Linear2DTermGenerator,
                    query_point,
                    &mut xtw_x,
                    &mut xtw_y,
                );
            }
            (3, PolynomialDegree::Linear) => {
                self.accumulate_normal_equations(
                    &weights,
                    Linear3DTermGenerator,
                    query_point,
                    &mut xtw_x,
                    &mut xtw_y,
                );
            }
            _ => {
                let term_gen =
                    GenericTermGenerator::new(self.polynomial_degree, self.dimensions, n_coeffs);
                self.accumulate_normal_equations(
                    &weights,
                    term_gen,
                    query_point,
                    &mut xtw_x,
                    &mut xtw_y,
                );
            }
        }

        // Column equilibration
        let mut col_norms = vec![T::one(); n_coeffs];
        for j in 0..n_coeffs {
            let diag = xtw_x[j * n_coeffs + j];
            if diag > T::epsilon() {
                col_norms[j] = diag.sqrt();
            }
        }
        for i in 0..n_coeffs {
            for j in 0..n_coeffs {
                let idx = i * n_coeffs + j;
                xtw_x[idx] = xtw_x[idx] / (col_norms[i] * col_norms[j]);
            }
            xtw_y[i] = xtw_y[i] / col_norms[i];
        }

        // Solve
        let beta = LinearSolver::solve_symmetric(&xtw_x, &xtw_y, n_coeffs);

        if let Some(mut coeffs) = beta {
            // Undo equilibration
            for j in 0..n_coeffs {
                coeffs[j] = coeffs[j] / col_norms[j];
            }
            // Return coeffs padded/truncated to d+1 elements
            let mut result = vec![T::zero(); d + 1];
            for (i, &c) in coeffs.iter().take(d + 1).enumerate() {
                result[i] = c;
            }
            return Some(result);
        }

        // Fallback: return [mean, 0, 0, ...]
        let (val, _) = self.weighted_mean_and_sum();
        let mut coeffs = vec![T::zero(); d + 1];
        coeffs[0] = val;
        Some(coeffs)
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

            let u = (dist / bandwidth).sqrt();
            let kernel_w = self.weight_function.compute_weight(u);

            let w = if self.use_robustness {
                (kernel_w * self.robustness_weights[idx]).sqrt()
            } else {
                kernel_w.sqrt()
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

    /// Internal WLS solver that uses the unified TermGenerator architecture.
    fn fit_polynomial_wls_internal(
        &self,
        weights: &[T],
        query_point: &[T],
        n_coeffs: usize,
        xtw_x: &mut [T],
        xtw_y: &mut [T],
    ) -> Option<(T, T)> {
        let n_neighbors = self.neighborhood.len();

        // Need at least as many neighbors as coefficients for a proper fit
        if n_neighbors < n_coeffs {
            // Compute weighted mean using provided weights
            let mut sum_w = T::zero();
            let mut sum_wy = T::zero();

            for (i, &w) in weights.iter().enumerate().take(n_neighbors) {
                let neighbor_idx = self.neighborhood.indices[i];
                let y_val = self.y[neighbor_idx];
                sum_w = sum_w + w;
                sum_wy = sum_wy + w * y_val;
            }

            let val = if sum_w > T::epsilon() {
                sum_wy / sum_w
            } else {
                // Fallback to simple mean
                let n = T::from(n_neighbors).unwrap();
                let mut sum = T::zero();
                for i in 0..n_neighbors {
                    let neighbor_idx = self.neighborhood.indices[i];
                    let y_val = self.y[neighbor_idx];
                    sum = sum + y_val;
                }
                sum / n
            };
            let leverage = if sum_w > T::epsilon() {
                T::one() / sum_w
            } else {
                T::zero()
            };
            return Some((val, leverage));
        }

        // 1. Accumulate Normal Equations
        match (self.dimensions, self.polynomial_degree) {
            (1, PolynomialDegree::Linear) => {
                self.accumulate_normal_equations(
                    weights,
                    Linear1DTermGenerator,
                    query_point,
                    xtw_x,
                    xtw_y,
                );
            }
            (2, PolynomialDegree::Linear) => {
                self.accumulate_normal_equations(
                    weights,
                    Linear2DTermGenerator,
                    query_point,
                    xtw_x,
                    xtw_y,
                );
            }
            (3, PolynomialDegree::Linear) => {
                self.accumulate_normal_equations(
                    weights,
                    Linear3DTermGenerator,
                    query_point,
                    xtw_x,
                    xtw_y,
                );
            }
            _ => {
                let term_gen =
                    GenericTermGenerator::new(self.polynomial_degree, self.dimensions, n_coeffs);
                self.accumulate_normal_equations(weights, term_gen, query_point, xtw_x, xtw_y);
            }
        }

        // 2. Column equilibration (normalize each column to unit norm)
        // For normal equations, we scale by sqrt(diagonal) to balance the matrix
        let mut col_norms = vec![T::one(); n_coeffs];
        for j in 0..n_coeffs {
            let diag = xtw_x[j * n_coeffs + j];
            if diag > T::epsilon() {
                col_norms[j] = diag.sqrt();
            }
        }

        // Apply equilibration: scale rows and columns of X'WX
        for i in 0..n_coeffs {
            for j in 0..n_coeffs {
                let idx = i * n_coeffs + j;
                xtw_x[idx] = xtw_x[idx] / (col_norms[i] * col_norms[j]);
            }
            // Also scale X'Wy
            xtw_y[i] = xtw_y[i] / col_norms[i];
        }

        // 3. Solve the equilibrated system
        let beta = LinearSolver::solve_symmetric(xtw_x, xtw_y, n_coeffs);

        if let Some(mut coeffs) = beta {
            // Undo equilibration on the solution
            for j in 0..n_coeffs {
                coeffs[j] = coeffs[j] / col_norms[j];
            }
            let leverage = if self.compute_leverage {
                // Solve A * x = e1 to get (A^-1)_00
                // Simply solve against [1, 0, ... 0]
                let mut e1 = vec![T::zero(); n_coeffs];
                e1[0] = T::one();

                LinearSolver::solve_symmetric(xtw_x, &e1, n_coeffs)
                    .map(|x| x[0])
                    .unwrap_or(T::zero())
            } else {
                T::zero()
            };

            return Some((coeffs[0], leverage));
        }

        // Fallback or Singularity Handling: use provided weights
        let mut sum_w = T::zero();
        let mut sum_wy = T::zero();
        let n_neighbors = self.neighborhood.len();

        for (i, &w) in weights.iter().enumerate().take(n_neighbors) {
            let neighbor_idx = self.neighborhood.indices[i];
            let y_val = self.y[neighbor_idx];
            sum_w = sum_w + w;
            sum_wy = sum_wy + w * y_val;
        }

        if sum_w > T::epsilon() {
            let val = sum_wy / sum_w;
            let leverage = T::one() / sum_w;
            Some((val, leverage))
        } else {
            self.handle_zero_weights()
        }
    }

    /// Helper to accumulate Normal Equations using a Generator.
    fn accumulate_normal_equations<G: TermGenerator<T>>(
        &self,
        weights: &[T],
        generator: G,
        query: &[T],
        xtwx: &mut [T],
        xtwy: &mut [T],
    ) {
        let n_coeffs = generator.n_coeffs();
        let n_neighbors = self.neighborhood.len();
        let d = self.dimensions;

        // Small stack buffer for terms
        const STACK_BUF_SIZE: usize = 64;
        let mut stack_terms = [T::zero(); STACK_BUF_SIZE];
        let mut heap_terms: Vec<T>;

        let terms_buf = if n_coeffs <= STACK_BUF_SIZE {
            &mut stack_terms[0..n_coeffs]
        } else {
            heap_terms = vec![T::zero(); n_coeffs];
            &mut heap_terms
        };

        // Clear accumulators
        for x in xtwx.iter_mut() {
            *x = T::zero();
        }
        for y in xtwy.iter_mut() {
            *y = T::zero();
        }

        for (i, &w) in weights.iter().enumerate().take(n_neighbors) {
            if w <= T::epsilon() {
                continue;
            }

            let neighbor_idx = self.neighborhood.indices[i];

            let offset = neighbor_idx * d;
            let (point, y_val) = (&self.x[offset..offset + d], self.y[neighbor_idx]);

            generator.generate(point, query, terms_buf);

            // Accumulate
            for j in 0..n_coeffs {
                let term_j = terms_buf[j];
                xtwy[j] = xtwy[j] + w * y_val * term_j;

                for (k, &term_k) in terms_buf.iter().enumerate().skip(j).take(n_coeffs - j) {
                    let idx = j * n_coeffs + k; // Row-major upper triangle index
                    xtwx[idx] = xtwx[idx] + w * term_j * term_k;
                }
            }
        }

        // Symmetrize
        for j in 0..n_coeffs {
            for k in j + 1..n_coeffs {
                let upper = j * n_coeffs + k;
                let lower = k * n_coeffs + j;
                xtwx[lower] = xtwx[upper];
            }
        }
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
