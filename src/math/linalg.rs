//! Linear algebra backend abstraction for LOESS.
//!
//! ## Purpose
//!
//! This module provides a trait-based abstraction over linear algebra operations,
//! standardizing on the optimized nalgebra backend.
//!
//! ## Design notes
//!
//! * Uses QR decomposition (Householder reflections) instead of Cholesky for better
//!   numerical stability with ill-conditioned systems.
//! * Fallback to SVD for rank-deficient matrices.
//! * Generic over `FloatLinalg` types (f32 and f64) which delegate to nalgebra.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use num_traits::Float;

// ============================================================================
// FloatLinalg Trait
// ============================================================================

/// Helper trait to bridge generic Float types to the optimized Nalgebra backend.
pub trait FloatLinalg: Float + 'static {
    /// Solve normal equations X'WX * beta = X'Wy.
    fn solve_normal(a: &[Self], b: &[Self], n: usize) -> Option<Vec<Self>>;
    /// Compute leverage (hat matrix diagonal element) for a query point.
    fn compute_leverage(design_vec: &[Self], xtw_x_inv: &[Self], n: usize) -> Self;
    /// Invert the normal matrix X'WX.
    fn invert_normal(a: &[Self], n: usize) -> Option<Vec<Self>>;
}

impl FloatLinalg for f64 {
    #[inline]
    fn solve_normal(a: &[Self], b: &[Self], n: usize) -> Option<Vec<Self>> {
        nalgebra_backend::solve_normal_equations_f64(a, b, n)
    }
    #[inline]
    fn compute_leverage(design_vec: &[Self], xtw_x_inv: &[Self], n: usize) -> Self {
        nalgebra_backend::compute_leverage_f64(design_vec, xtw_x_inv, n)
    }
    #[inline]
    fn invert_normal(a: &[Self], n: usize) -> Option<Vec<Self>> {
        nalgebra_backend::invert_normal_matrix_f64(a, n)
    }
}

impl FloatLinalg for f32 {
    #[inline]
    fn solve_normal(a: &[Self], b: &[Self], n: usize) -> Option<Vec<Self>> {
        nalgebra_backend::solve_normal_equations_f32(a, b, n)
    }
    #[inline]
    fn compute_leverage(design_vec: &[Self], xtw_x_inv: &[Self], n: usize) -> Self {
        nalgebra_backend::compute_leverage_f32(design_vec, xtw_x_inv, n)
    }
    #[inline]
    fn invert_normal(a: &[Self], n: usize) -> Option<Vec<Self>> {
        nalgebra_backend::invert_normal_matrix_f32(a, n)
    }
}

// ============================================================================
// Nalgebra Backend Implementation
// ============================================================================

/// Nalgebra-based linear algebra operations.
pub mod nalgebra_backend {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// Solve normal equations X'WX * beta = X'Wy using f64 precision.
    pub fn solve_normal_equations_f64(
        xtw_x: &[f64],
        xtw_y: &[f64],
        n_coeffs: usize,
    ) -> Option<Vec<f64>> {
        let matrix = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x);
        let rhs = DVector::from_column_slice(xtw_y);

        let qr = matrix.clone().qr();
        if let Some(solution) = qr.solve(&rhs) {
            return Some(solution.as_slice().to_vec());
        }

        matrix
            .svd(true, true)
            .solve(&rhs, f64::EPSILON * 100.0)
            .ok()
            .map(|s: DVector<f64>| s.as_slice().to_vec())
    }

    /// Compute leverage for a query point using f64 precision.
    pub fn compute_leverage_f64(design_vec: &[f64], xtw_x_inv: &[f64], n_coeffs: usize) -> f64 {
        let x = DVector::from_column_slice(design_vec);
        let inv = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x_inv);
        (x.transpose() * &inv * &x)[(0, 0)]
    }

    /// Invert normal matrix X'WX using f64 precision.
    pub fn invert_normal_matrix_f64(xtw_x: &[f64], n_coeffs: usize) -> Option<Vec<f64>> {
        let matrix = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x);
        let qr = matrix.clone().qr();
        let identity = DMatrix::identity(n_coeffs, n_coeffs);

        if let Some(inv) = qr.solve(&identity) {
            return Some(inv.as_slice().to_vec());
        }

        matrix
            .pseudo_inverse(f64::EPSILON * 100.0)
            .ok()
            .map(|inv: DMatrix<f64>| inv.as_slice().to_vec())
    }

    /// Solve normal equations X'WX * beta = X'Wy using f32 precision.
    pub fn solve_normal_equations_f32(
        xtw_x: &[f32],
        xtw_y: &[f32],
        n_coeffs: usize,
    ) -> Option<Vec<f32>> {
        let matrix = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x);
        let rhs = DVector::from_column_slice(xtw_y);

        let qr = matrix.clone().qr();
        if let Some(solution) = qr.solve(&rhs) {
            return Some(solution.as_slice().to_vec());
        }

        matrix
            .svd(true, true)
            .solve(&rhs, f32::EPSILON * 100.0)
            .ok()
            .map(|s: DVector<f32>| s.as_slice().to_vec())
    }

    /// Compute leverage for a query point using f32 precision.
    pub fn compute_leverage_f32(design_vec: &[f32], xtw_x_inv: &[f32], n_coeffs: usize) -> f32 {
        let x = DVector::from_column_slice(design_vec);
        let inv = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x_inv);
        (x.transpose() * &inv * &x)[(0, 0)]
    }

    /// Invert normal matrix X'WX using f32 precision.
    pub fn invert_normal_matrix_f32(xtw_x: &[f32], n_coeffs: usize) -> Option<Vec<f32>> {
        let matrix = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x);
        let qr = matrix.clone().qr();
        let identity = DMatrix::identity(n_coeffs, n_coeffs);

        if let Some(inv) = qr.solve(&identity) {
            return Some(inv.as_slice().to_vec());
        }

        matrix
            .pseudo_inverse(f32::EPSILON * 100.0)
            .ok()
            .map(|inv: DMatrix<f32>| inv.as_slice().to_vec())
    }
}
