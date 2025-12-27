//! Interpolation and delta optimization for LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides utilities for optimized LOESS performance through
//! delta-based point skipping and linear interpolation. When data points are
//! densely sampled, fitting every point is computationally expensive and
//! often unnecessary.
//!
//! ## Design notes
//!
//! * **Optimization**: Delta controls the distance threshold for skipping.
//! * **Interpolation**: Uses linear interpolation to fill gaps between fitted anchors.
//! * **Defaults**: Calculate conservative default delta (1% of range) if needed.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Delta Optimization**: Fits anchor points spaced at least `delta` apart.
//! * **Linear Interpolation**: Fills gaps: y = y_0 + alpha * (y_1 - y_0).
//! * **Tied Values**: Tied x-values receive the same fitted value.
//!
//! ## Invariants
//!
//! * Input x-values must be sorted in ascending order.
//! * Delta must be non-negative and finite.
//! * At least one point is always fitted.
//!
//! ## Non-goals
//!
//! * This module does not perform the actual smoothing/fitting.
//! * This module does not sort the input data.
//! * This module does not provide higher-order interpolation.

// External dependencies
use core::result::Result;
use num_traits::Float;

// Internal dependencies
use crate::primitives::errors::LoessError;

// ============================================================================
// Delta Calculation
// ============================================================================

/// Calculate delta parameter for interpolation optimization.
///
/// # Default behavior
///
/// If delta is `None`, computes a conservative default as 1% of the x-range:
/// ```text
/// delta = 0.01 × (max(x) - min(x))
/// ```
pub fn calculate_delta<T: Float>(delta: Option<T>, x_sorted: &[T]) -> Result<T, LoessError> {
    match delta {
        Some(d) => Ok(d),
        None => {
            // Compute default delta as 1% of x-range
            if x_sorted.is_empty() {
                Ok(T::zero())
            } else {
                let range = x_sorted[x_sorted.len() - 1] - x_sorted[0];
                Ok(T::from(0.01).unwrap() * range)
            }
        }
    }
}
