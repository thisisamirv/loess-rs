//! Distance metrics for nD LOESS.
//!
//! ## Purpose
//!
//! This module provides distance computation for multivariate (nD) LOESS.
//! In 1D LOWESS, distance is simply |x - x_i|. For nD LOESS, we need proper
//! distance metrics that work in higher-dimensional spaces.
//!
//! ## Design notes
//!
//! * **Decoupling**: Distance calculation is separated from kernel evaluation.
//! * **Normalization**: Supports normalizing dimensions to handle differing scales.
//!
//! ## Key concepts
//!
//! * **Metric**: Defines how "closeness" is measured (Euclidean, Manhattan, etc.).
//! * **Normalization**: Rescaling dimensions to [0, 1] to ensure equal influence.
//!
//! ## Invariants
//!
//! * Distance is always non-negative.
//! * Distance is zero if and only if points are identical (for metrics satisfying identity).
//!
//! ## Non-goals
//!
//! * This module does not handle the kernel weighting (bandwidth/smoothing).

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering;
use num_traits::Float;

// ============================================================================
// Distance Metric Enum
// ============================================================================

/// Distance metric for nD LOESS neighborhood computation.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DistanceMetric<T> {
    /// Standard Euclidean distance: √(Σ(xᵢ - yᵢ)²)
    Euclidean,

    /// Normalized Euclidean distance.
    #[default]
    Normalized,

    /// Manhattan distance (L1 norm): Σ|xᵢ - yᵢ|
    Manhattan,

    /// Chebyshev distance (L∞ norm): max|xᵢ - yᵢ|
    Chebyshev,

    /// Minkowski distance (Lp norm): (Σ|xᵢ - yᵢ|^p)^(1/p)
    /// Generalization of Euclidean (p=2) and Manhattan (p=1).
    /// p must be >= 1.0.
    Minkowski(T),

    /// Weighted Euclidean distance: √(Σ wᵢ(xᵢ - yᵢ)²)
    Weighted(Vec<T>),
}

// ============================================================================
// Distance Computation Functions
// ============================================================================

impl<T: Float> DistanceMetric<T> {
    /// Compute Euclidean distance between two nD points.
    #[inline]
    pub fn euclidean(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| {
                let diff = ai - bi;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt()
    }

    /// Compute normalized distance between two nD points.
    #[inline]
    pub fn normalized(a: &[T], b: &[T], scales: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), scales.len());

        let sum_sq = a
            .iter()
            .zip(b.iter())
            .zip(scales.iter())
            .map(|((&ai, &bi), &scale)| {
                let diff = (ai - bi) * scale;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);

        sum_sq.sqrt()
    }

    /// Compute Manhattan distance (L1 norm).
    #[inline]
    pub fn manhattan(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs())
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Compute Chebyshev distance (L-inf norm).
    #[inline]
    pub fn chebyshev(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs())
            .fold(T::zero(), T::max)
    }

    /// Compute Minkowski distance (Lp norm).
    #[inline]
    pub fn minkowski(a: &[T], b: &[T], p: T) -> T {
        debug_assert_eq!(a.len(), b.len());
        let sum_pow = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs().powf(p))
            .fold(T::zero(), |acc, x| acc + x);
        sum_pow.powf(T::one() / p)
    }

    /// Compute Weighted Euclidean distance.
    #[inline]
    pub fn weighted(a: &[T], b: &[T], weights: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), weights.len());
        let sum_sq = a
            .iter()
            .zip(b.iter())
            .zip(weights.iter())
            .map(|((&ai, &bi), &w)| {
                let diff = ai - bi;
                w * diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);
        sum_sq.sqrt()
    }

    /// Compute robust normalization scales using 10% trimmed standard deviation.
    pub fn compute_robust_scales(points: &[T], dimensions: usize) -> Vec<T> {
        let n = points.len() / dimensions;
        if n < 10 {
            // Fallback for very small datasets: use simple range or 1.0
            return vec![T::one(); dimensions];
        }

        let cut = (T::from(0.1).unwrap() * T::from(n).unwrap())
            .ceil()
            .to_usize()
            .unwrap();
        // Ensure we don't trim everything
        let effective_n = n.saturating_sub(2 * cut);
        if effective_n < 2 {
            return vec![T::one(); dimensions];
        }

        let mut scales = Vec::with_capacity(dimensions);
        let mut temp = Vec::with_capacity(n);

        for d in 0..dimensions {
            // Extract dimension d
            temp.clear();
            for i in 0..n {
                temp.push(points[i * dimensions + d]);
            }

            // Sort
            temp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            // Compute trimmed mean
            let sum: T = temp
                .iter()
                .skip(cut)
                .take(effective_n)
                .fold(T::zero(), |acc, &val| acc + val);
            let mean = sum / T::from(effective_n).unwrap();

            // Compute trimmed sum of squares
            let sum_sq: T = temp
                .iter()
                .skip(cut)
                .take(effective_n)
                .map(|&val| {
                    let diff = val - mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, val| acc + val);

            // Compute standard deviation (divisor = N_eff - 1)
            let variance = sum_sq / T::from(effective_n - 1).unwrap();
            let std_dev = variance.sqrt();

            if std_dev > T::epsilon() {
                scales.push(T::one() / std_dev);
            } else {
                scales.push(T::zero()); // Ignore constant dimension
            }
        }
        scales
    }
}
